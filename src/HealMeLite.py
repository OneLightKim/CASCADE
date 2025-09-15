import os
import json
import argparse
from tqdm import tqdm
from attrdict import AttrDict
from openai import OpenAI



# --------------------------------------------------------------------------------
# 1. ChatGPT API 호출을 위한 함수 (첫 번째 스크립트에서 가져옴)
# --------------------------------------------------------------------------------
def create_chat_completion(client, system_prompt, user_prompt, model="gpt-3.5-turbo", temperature=0.8, max_tokens=650):
    """
    OpenAI의 Chat Completion API를 호출하는 함수.
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        # 응답이 유효한지 확인 후 content 반환
        if response and response.choices:
            return response.choices[0].message.content.strip()
        else:
            print("Warning: Invalid response structure from API.")
            return "Error: Invalid response from API."
    except Exception as e:
        print(f"Error during API call: {e}")
        return f"Error: {str(e)}"

# --------------------------------------------------------------------------------
# 2. 제로샷 Cognitive Reframing 상담 시스템
# --------------------------------------------------------------------------------

# --------------------------------------------------------------------------------
# 3. 메인 파이프라인 함수 (제로샷 Cognitive Reframing)
# --------------------------------------------------------------------------------
def main(cli_args):
    try:
        args = AttrDict(vars(cli_args))
        
        # OpenAI 클라이언트 초기화
        client = OpenAI(api_key=args.api_key)
        
        # PsyQA 번역 데이터 로드
        print("Loading PsyQA dataset...")
        with open(args.data_file, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
        
        print(f"전체 데이터셋 크기: {len(dataset)}")
        print("@@ Cognitive Reframing 제로샷 방식으로 처리합니다 (분류 모델 없음)")

        # 출력 디렉토리 생성
        os.makedirs(args.output_dir, exist_ok=True)
        output_path = os.path.join(args.output_dir, args.output_file)

        # 기존 결과 파일이 있는지 확인하여 중간부터 재시작
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
                print("처음부터 다시 시작합니다.")
                processed_ids = set()
                start_idx = 0

        # 파일 모드 결정 (append 또는 write)
        file_mode = 'a' if start_idx > 0 else 'w'
        
        with open(output_path, file_mode, encoding='utf-8') as of:
            remaining_dataset = dataset[start_idx:] if start_idx > 0 else dataset
            progress_desc = f"Processing samples ({start_idx+1}-{len(dataset)}) with Cognitive Reframing" if start_idx > 0 else f"Processing {len(dataset)} samples with Cognitive Reframing"
            
            for relative_idx, data in enumerate(tqdm(remaining_dataset, desc=progress_desc, initial=start_idx, total=len(dataset))):
                try:
                    # 실제 인덱스 계산 (전체 데이터셋에서의 위치)
                    actual_idx = start_idx + relative_idx
                    
                    # PsyQA 데이터 구조에 맞게 필드 추출
                    question_id = data.get("idx", "")  # 새로 추가된 idx 사용
                    
                    # 이미 처리된 ID인지 확인 (중복 방지)
                    if question_id in processed_ids:
                        print(f"@@ 이미 처리된 샘플 건너뜀: {question_id}")
                        continue
                    
                    question_title = data.get("question", "")  # 번역된 질문
                    question_desc = data.get("description", "")  # 번역된 설명
                    answers = data.get("answers", [])
                    answer_text = answers[0].get("answer_text", "") if answers else ""

                    # 질문과 설명을 합쳐서 전체 텍스트 생성
                    post = f"{question_title}. {question_desc}"

                    # Cognitive Reframing 프롬프트 생성 (제로샷 방식)
                    system_prompt = (
                        "You are a psychotherapist practicing the 'cognitive reframing' strategy to help clients reframe their negative emotions. "
                        "The client will provide a single statement expressing their negative thoughts and feelings. "
                        "Please provide a comprehensive, cohesive, and empathetic response of approximately 650 tokens that flows naturally, avoids list formats, and maintains an emotionally supportive tone throughout."
                    )

                    user_prompt = (
                        "Your task is to: "
                        "1. Identify and separate the situation from the client's thoughts and emotions. "
                        "2. Suggest alternative, more constructive perspectives they could consider. "
                        "3. Provide a concise, empathetic, and supportive response that combines these reframed thoughts with encouragement. "
                        f"The client's statement: {post}"
                    )
                    print(f"@@ [{actual_idx+1}/{len(dataset)}] Cognitive Reframing 상담 응답 생성 중...")
                    
                    # ChatGPT API 호출로 텍스트 생성
                    generated = create_chat_completion(client, system_prompt, user_prompt, model="gpt-3.5-turbo", max_tokens=650)

                    # 결과 저장 (제로샷 방식)
                    result = {
                        "questionID": question_id,
                        "total_questionText": post.replace('\n', ' ').replace('\r', ' '),
                        "model_answer": generated.replace('\n', ' ').replace('\r', ' '),
                        "answerText": answer_text.replace('\n', ' ').replace('\r', ' '),
                        "approach": "cognitive_reframing_zero_shot"
                            }
                    of.write(json.dumps(result, ensure_ascii=False) + "\n")
                    of.flush()

                except Exception as e:
                    print(f"X Error processing data for questionID {question_id}: {str(e)}")
                    of.flush()
                    continue

        print(f"Results successfully saved to {output_path}")

    except Exception as e:
        print(f"An error occurred in the main execution: {str(e)}")
        raise

if __name__ == "__main__":
    cli_parser = argparse.ArgumentParser(description="Run cognitive reframing counseling using RoBERTa classification and GPT-4o zero-shot generation.")
    
    # 필수 인자
    cli_parser.add_argument("--api_key", type=str, default="", help="Your OpenAI API key.")
    
    # 파일 경로 인자
    cli_parser.add_argument("--data_file", type=str, default="data/main/400_psyqa_translated.json", help="Path to the 400-sample PsyQA data file.")
    cli_parser.add_argument("--output_dir", type=str, default="./output_dir/pipeline", help="Directory to save the output file.")
    cli_parser.add_argument("--output_file", type=str, default="healmelite_gpt35_psyqa_inference.jsonl", help="Name of the output file.")
    
    args = cli_parser.parse_args()
    
    main(args)