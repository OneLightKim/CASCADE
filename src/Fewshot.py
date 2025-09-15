import os
import json
import argparse
from tqdm import tqdm
from attrdict import AttrDict
from openai import OpenAI
import random

# --------------------------------------------------------------------------------
# 1. ChatGPT API 호출을 위한 함수
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
        if response and response.choices:
            return response.choices[0].message.content.strip()
        else:
            print("Warning: Invalid response structure from API.")
            return "Error: Invalid response from API."
    except Exception as e:
        print(f"Error during API call: {e}")
        return f"Error: {str(e)}"

# --------------------------------------------------------------------------------
# 2. Few-shot 예제 관리를 위한 함수들
# --------------------------------------------------------------------------------
def select_examples(example_pool, current_data, n_examples=2):
    """
    data/main/400_psyqa_translated.json에서 현재 케이스를 제외하고 n개를 랜덤하게 선택
    Args:
        example_pool: 예제 풀 데이터셋 (400_psyqa_translated.json)
        current_data: 현재 처리 중인 데이터 항목 (딕셔너리)
        n_examples: 선택할 예제 수
    """
    # 현재 케이스를 제외한 다른 예제들 필터링
    other_examples = []
    current_question = current_data.get("question", "")
    current_idx = current_data.get("idx", "")
    
    for data in example_pool:
        # 현재 케이스와 다른 예제만 선택 (idx로 비교, 없으면 질문 텍스트로 비교)
        is_different_case = (
            (current_idx and data.get("idx") != current_idx) or 
            (not current_idx and data.get("question") != current_question)
        )
        
        if (is_different_case and 
            data.get("question") and  # 질문이 있는지 확인
            data.get("answers") and  # 답변이 있는지 확인
            len(data["answers"]) > 0 and  # 답변이 최소 1개 이상인지 확인
            data["answers"][0].get("answer_text")):  # 첫 번째 답변에 텍스트가 있는지 확인
            
            # question과 description을 합친 full question 생성
            full_question = data["question"]
            if data.get("description"):
                full_question += f". {data['description']}"
            
            example = {
                "question": full_question,
                "answer": data["answers"][0]["answer_text"]
            }
            other_examples.append(example)
    
    # 사용 가능한 예제 수 확인
    if len(other_examples) < n_examples:
        print(f"Warning: Only {len(other_examples)} examples available")
        return other_examples
    
    return random.sample(other_examples, n_examples)

def format_examples(examples):
    """
    선택된 예제들을 프롬프트에 맞는 형식으로 포매팅
    """
    formatted = ""
    for i, ex in enumerate(examples, 1):
        formatted += f"Example {i}:\n"
        formatted += f"User: {ex['question']}\n"
        formatted += f"Counselor: {ex['answer']}\n\n"
    return formatted

# --------------------------------------------------------------------------------
# 3. 메인 파이프라인 함수
# --------------------------------------------------------------------------------
def main(cli_args):
    try:
        args = AttrDict(vars(cli_args))
        
        # OpenAI 클라이언트 초기화
        client = OpenAI(api_key=args.api_key)

        # PsyQA 번역 데이터 로드
        print("Loading translated PsyQA dataset...")
        with open(args.data_file, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
        
        # Few-shot 예제 풀 로드 (400_psyqa_translated.json)
        print("Loading few-shot example pool...")
        with open("data/main/400_psyqa_translated.json", 'r', encoding='utf-8') as f:
            example_pool = json.load(f)
        
        print(f"Loaded {len(dataset)} questions from dataset")
        print(f"Loaded {len(example_pool)} examples for few-shot prompting")

        # 출력 디렉토리 생성
        os.makedirs(args.output_dir, exist_ok=True)
        output_path = os.path.join(args.output_dir, args.output_file)

        with open(output_path, 'w', encoding='utf-8') as of:
            for data in tqdm(dataset, desc="Inference with ChatGPT"):
                try:
                    # PsyQA 데이터 구조에 맞게 필드 추출
                    question_title = data.get("question", "")
                    question_desc = data.get("description", "")
                    answers = data.get("answers", [])
                    answer_text = answers[0].get("answer_text", "") if answers else ""

                    # 질문과 설명을 합쳐서 전체 텍스트 생성
                    post = f"{question_title}. {question_desc}"

                    # Few-shot 예제 선택 및 포매팅 (400_psyqa_translated.json에서 무작위 선택)
                    selected_examples = select_examples(example_pool, data, n_examples=2)
                    examples_text = format_examples(selected_examples)

                    # 프롬프트 생성
                    system_prompt = (
                        "You are a mental health counseling expert. Provide responses that help users address their mental health concerns. "
                        "Please provide a comprehensive, cohesive, and empathetic response of approximately 650 tokens that flows naturally, avoids list formats, and maintains an emotionally supportive tone throughout."
                    )
                    user_prompt = (
                        f"The user's counseling content is as follows:\n\n{post}\n\n"
                        "Here are some example responses for similar cases:\n\n"
                        f"{examples_text}\n"
                        "Please provide a counseling response for the user's case."
                    )

                    # ChatGPT API 호출로 텍스트 생성
                    generated = create_chat_completion(client, system_prompt, user_prompt, model="gpt-3.5-turbo", max_tokens=650)

                    # 결과 저장
                    result = {
                        "questionID": data.get("id", ""),
                        "total_questionText": post.replace('\n', ' ').replace('\r', ' '),
                        "model_answer": generated.replace('\n', ' ').replace('\r', ' '),
                        "answerText": answer_text.replace('\n', ' ').replace('\r', ' '),
                        "few_shot_examples": [{"question": ex["question"], "answer": ex["answer"]} for ex in selected_examples]
                    }
                    of.write(json.dumps(result, ensure_ascii=False) + "\n")

                except Exception as e:
                    print(f"Error processing data for question: {question_title[:50]}...: {str(e)}")
                    continue

        print(f"Results successfully saved to {output_path}")

    except Exception as e:
        print(f"An error occurred in the main execution: {str(e)}")
        raise

if __name__ == "__main__":
    cli_parser = argparse.ArgumentParser(description="Run few-shot inference using ChatGPT for generation.")
    
    # 필수 인자
    cli_parser.add_argument("--api_key", type=str, default="", help="Your OpenAI API key.")
    
    # 파일 경로 인자
    cli_parser.add_argument("--data_file", type=str, default="data/main/400_psyqa_translated.json", help="Path to the translated PsyQA data file.")
    cli_parser.add_argument("--output_dir", type=str, default="./output_dir/pipeline", help="Directory to save the output file.")
    cli_parser.add_argument("--output_file", type=str, default="fewshot_gpt35_psyqa_inference.jsonl", help="Name of the output file.")
    
    args = cli_parser.parse_args()
    main(args)
