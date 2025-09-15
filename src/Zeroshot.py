import os
import json
import argparse
from tqdm import tqdm
from attrdict import AttrDict
from openai import OpenAI

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
# 2. 메인 파이프라인 함수
# --------------------------------------------------------------------------------
def main(cli_args):
    try:
        args = AttrDict(vars(cli_args))
        
        # OpenAI 클라이언트 초기화
        client = OpenAI(api_key=args.api_key)
        
        # PsyQA 번역 데이터 로드 (처음 1000개만)
        print("Loading translated PsyQA dataset...")
        with open(args.data_file, 'r', encoding='utf-8') as f:
            full_dataset = json.load(f)
            dataset = full_dataset[:1000]  # 처음 1000개만 선택
        
        print(f"Loaded {len(dataset)} questions from translated PsyQA dataset")

        # 출력 디렉토리 생성
        os.makedirs(args.output_dir, exist_ok=True)
        output_path = os.path.join(args.output_dir, args.output_file)

        with open(output_path, 'w', encoding='utf-8') as of:
            for data in tqdm(dataset, desc="Inference with ChatGPT"):
                try:
                    # PsyQA 데이터 구조에 맞게 필드 추출
                    question_id = data.get("id", "")
                    question_title = data.get("question", "")  # 번역된 질문
                    question_desc = data.get("description", "")  # 번역된 설명
                    answers = data.get("answers", [])
                    answer_text = answers[0].get("answer_text", "") if answers else ""

                    # 질문과 설명을 합쳐서 전체 텍스트 생성
                    post = f"{question_title}. {question_desc}"

                    # 1. 프롬프트 생성
                    system_prompt = (
                        "You are a mental health counseling expert. Provide responses that help users address their mental health concerns. "
                        "Please provide a comprehensive, cohesive, and empathetic response of approximately 650 tokens that flows naturally, avoids list formats, and maintains an emotionally supportive tone throughout."
                    )
                    user_prompt = f"The user's counseling content is as follows:\n\n{post}\n\nPlease provide a counseling response."

                    # 2. ChatGPT API 호출로 텍스트 생성
                    generated = create_chat_completion(client, system_prompt, user_prompt, model="gpt-3.5-turbo", max_tokens=650)

                    # 3. 결과 저장
                    result = {
                        "questionID": question_id,
                        "total_questionText": post,
                        "model_answer": generated,
                        "answerText": answer_text
                    }
                    of.write(json.dumps(result, ensure_ascii=False) + "\n")

                except Exception as e:
                    print(f"Error processing data for questionID {data.get('id', 'unknown')}: {str(e)}")
                    continue

        print(f"Results successfully saved to {output_path}")

    except Exception as e:
        print(f"An error occurred in the main execution: {str(e)}")
        raise

if __name__ == "__main__":
    cli_parser = argparse.ArgumentParser(
        description="Run inference using ChatGPT for generation."
    )
    
    # 필수 인자
    cli_parser.add_argument("--api_key", type=str, default="", help="Your OpenAI API key.")
    
    # 파일 경로 인자
    cli_parser.add_argument("--data_file", type=str, default="data/main/400_psyqa_translated.json", help="Path to the translated PsyQA data file.")
    cli_parser.add_argument("--output_dir", type=str, default="./output_dir/pipeline", help="Directory to save the output file.")
    cli_parser.add_argument("--output_file", type=str, default="zeroshot_gpt35_psyqa_inference.jsonl", help="Name of the output file.")
    
    args = cli_parser.parse_args()
    main(args)
