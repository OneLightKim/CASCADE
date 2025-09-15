# src/CBTLLM.py
import os
import json
import argparse
from tqdm import tqdm
from attrdict import AttrDict
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# CBT-LLM 관련 설정
CBT_MODEL_NAME = "Hongbin37/CBT-LLM"


# --------------------------------------------------------------------------------
# 1. CBT-LLM 모델 로드 및 추론 함수
# --------------------------------------------------------------------------------
def load_cbt_model():
    """
    CBT-LLM 모델과 토크나이저를 로드 (Auto* + trust_remote_code=True).
    Baichuan 전용 클래스를 직접 임포트하지 않고, 한 저장소만 신뢰해
    config_class mismatch를 방지.
    """
    print(f"@@ CBT-LLM 모델 로딩 중: {CBT_MODEL_NAME}")

    # 토크나이저
    tokenizer = AutoTokenizer.from_pretrained(
        CBT_MODEL_NAME,
        trust_remote_code=True
    )

    # 모델
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        CBT_MODEL_NAME,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        device_map="auto" if torch.cuda.is_available() else None,
        # 필요하면 특정 커밋/태그로 고정:
        # revision="main",
    )

    # pad 토큰 보정 (없으면 eos로 대체)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval()
    print(f"@@ CBT-LLM 모델 로딩 완료 (device: {device})")
    return model, tokenizer, device


def generate_cbt_response(model, tokenizer, device, question, description):
    """
    CBT-LLM을 사용하여 CBT 기반 응답을 생성하는 함수.
    """
    try:
        # CBT 프롬프트 템플릿
        cbt_prompt = (
            "Based on the following question and its description, please provide a professional, compassionate, and helpful response. "
            "Ensure your response adheres to the structure of Cognitive Behavioral Therapy (CBT) responses, especially in identifying the key thought or belief, and seamlessly integrates each part:\n\n"
            "1. Validation and Empathy: Show understanding and sympathy for the patient's feelings or issues, creating a sense of safety.\n"
            "2. Identify Key Thought or Belief: Through the problem description, identify potential cognitive distortions or core beliefs.\n"
            "3. Pose Challenge or Reflection: Raise open-ended questions, encouraging the patient to reconsider or reflect on their initial thoughts or beliefs.\n"
            "4. Provide Strategy or Insight: Offer practical strategies or insights to help them deal with the current situation.\n"
            "5. Encouragement and Foresight: Encourage the patient to use the strategy, emphasizing that this is just the beginning and further support may be needed.\n\n"
            f"Question: {question}\n"
            f"Description: {description}\n"
            "Response:"
        )

        # CBT-LLM 템플릿 (원 코드 형식 유지)
        template = (
            "You are an experienced psychological counselor, specialized in Cognitive Behavioral Therapy (CBT). "
            "Please respond to the following question as a CBT counselor.\n"
            "Human: {}\nAssistant: "
        )
        formatted_prompt = template.format(cbt_prompt)

        inputs = tokenizer([formatted_prompt], return_tensors="pt", padding=True)
        if device == "cuda":
            inputs = {k: v.cuda() for k, v in inputs.items()}

        with torch.no_grad():
            generate_ids = model.generate(
                **inputs,
                max_new_tokens=650,
                temperature=0.8,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )

        response = tokenizer.decode(generate_ids[0], skip_special_tokens=True)
        if "Assistant: " in response:
            response = response.split("Assistant: ")[-1].strip()

        return response

    except Exception as e:
        print(f"CBT 응답 생성 중 오류: {e}")
        return f"Error generating CBT response: {str(e)}"


# --------------------------------------------------------------------------------
# 2. 메인 파이프라인 함수 (CBT-LLM 사용)
# --------------------------------------------------------------------------------
def main(cli_args):
    try:
        args = AttrDict(vars(cli_args))

        # CBT-LLM 모델 로딩
        cbt_model, cbt_tokenizer, device = load_cbt_model()

        # PsyQA 번역 데이터 로드
        print("Loading 400-sample PsyQA dataset...")
        with open(args.data_file, 'r', encoding='utf-8') as f:
            full_dataset = json.load(f)

        dataset = full_dataset
        print(f"전체 데이터셋 크기: {len(full_dataset)}")
        print(f"처리할 데이터 크기: {len(dataset)} (400개 샘플 예상)")

        # 출력 경로
        os.makedirs(args.output_dir, exist_ok=True)
        output_path = os.path.join(args.output_dir, args.output_file)

        # 중간 재시작 처리
        processed_ids = set()
        start_idx = 0

        if os.path.exists(output_path):
            print(f"@@ 기존 결과 파일 발견: {output_path}")
            try:
                with open(output_path, 'r', encoding='utf-8') as existing_file:
                    for line_num, line in enumerate(existing_file):
                        if line.strip():
                            try:
                                result = json.loads(line.strip())
                                processed_ids.add(result.get("questionID", ""))
                            except json.JSONDecodeError:
                                print(f"!! 줄 {line_num+1}에서 JSON 파싱 오류, 무시합니다.")
                                continue
                start_idx = len(processed_ids)
                print(f"@@ 이미 처리된 샘플: {len(processed_ids)}개")
                print(f"@@ 인덱스 {start_idx}부터 재시작합니다.")
            except Exception as e:
                print(f"!! 기존 파일 읽기 오류: {e}")
                processed_ids = set()
                start_idx = 0

        file_mode = 'a' if start_idx > 0 else 'w'

        with open(output_path, file_mode, encoding='utf-8') as of:
            remaining_dataset = dataset[start_idx:] if start_idx > 0 else dataset
            total_n = len(dataset)
            desc = (
                f"Processing samples ({start_idx+1}-{total_n}) with CBT-LLM"
                if start_idx > 0 else f"Processing {total_n} samples with CBT-LLM"
            )

            for relative_idx, data in enumerate(tqdm(remaining_dataset, desc=desc, total=len(remaining_dataset))):
                try:
                    actual_idx = start_idx + relative_idx

                    # PsyQA 필드
                    question_id = data.get("idx", "")
                    if question_id in processed_ids:
                        # 중복 방지
                        continue

                    question_title = data.get("question", "")
                    question_desc = data.get("description", "")
                    answers = data.get("answers", [])
                    answer_text = answers[0].get("answer_text", "") if answers else ""

                    # CBT-LLM 응답 생성
                    generated = generate_cbt_response(
                        cbt_model,
                        cbt_tokenizer,
                        device,
                        question_title,
                        question_desc
                    )

                    full_question = f"{question_title}. {question_desc}"
                    result = {
                        "questionID": question_id,
                        "question": question_title,
                        "description": question_desc,
                        "total_questionText": full_question.replace('\n', ' ').replace('\r', ' '),
                        "cbt_response": generated.replace('\n', ' ').replace('\r', ' '),
                        "answerText": answer_text.replace('\n', ' ').replace('\r', ' '),
                        "model_used": "CBT-LLM"
                    }

                    of.write(json.dumps(result, ensure_ascii=False) + "\n")
                    of.flush()

                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                except Exception as e:
                    print(f"X Error processing data for questionID {data.get('idx','')}: {str(e)}")
                    of.flush()
                    continue

        print(f"Results successfully saved to {output_path}")

    except Exception as e:
        print(f"An error occurred in the main execution: {str(e)}")
        raise


if __name__ == "__main__":
    cli_parser = argparse.ArgumentParser(
        description="Run inference using CBT-LLM for cognitive behavioral therapy responses."
    )
    cli_parser.add_argument(
        "--data_file",
        type=str,
        default="data/main/400_psyqa_translated.json",
        help="Path to the 400-sample PsyQA data file."
    )
    cli_parser.add_argument(
        "--output_dir",
        type=str,
        default="./output_dir/pipeline",
        help="Directory to save the output file."
    )
    cli_parser.add_argument(
        "--output_file",
        type=str,
        default="CBTLLM_psyqa_inference.jsonl",
        help="Name of the output file."
    )
    args = cli_parser.parse_args()
    main(args)
