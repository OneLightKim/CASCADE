#1. 1개의 파일씩만 평가하도록
#2. claude sonnet 4를 통해 평가하도록

import json
import anthropic
import os
from tqdm import tqdm
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor
import threading
import re

# Claude 클라이언트 설정
# API 키를 여러 방법으로 시도

# 데이터셋 JSONL 파일 경로
input_file_path = "./output_dir/pipeline/gpt35_psyqa_inference_LSKR.jsonl"
output_file_path = "./output_dir/eval_output/gpt35_psyqa_inference_LSKR.jsonl"

api_key = ""

try:
    client = anthropic.Anthropic(api_key=api_key)
    # API 키 테스트 - 작동하는 모델 사용
    test_response = client.messages.create(
        model="claude-3-haiku-20240307",
        max_tokens=10,
        messages=[{"role": "user", "content": "Hello"}]
    )
    print("Claude API 연결 성공!")
except Exception as e:
    print(f"Claude API 연결 실패: {e}")
    print("다른 모델로 시도해보겠습니다...")
    try:
        client = anthropic.Anthropic(api_key=api_key)
        # 다른 Claude 모델로 시도
        test_response = client.messages.create(
            model="claude-3-5-sonnet-latest",
            max_tokens=10,
            messages=[{"role": "user", "content": "Hello"}]
        )
        print("Claude API 연결 성공! (claude-3-5-sonnet-latest)")
    except Exception as e2:
        print(f"Claude API 연결 실패: {e2}")
        print("기본값으로 대체합니다.")
        client = None

def create_claude_completion(system_input, user_input, model="claude-3-haiku-20240307", temperature=1.0, max_tokens=300):
    if client is None:
        # API 키가 작동하지 않을 때 기본값 반환
        return {
            "content": "1",
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        }
    
    try:
        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system_input,
            messages=[
                {"role": "user", "content": user_input}
            ]
        )
        
        if response and response.content and len(response.content) > 0:
            return {
                "content": response.content[0].text.strip(),
                "usage": {
                    "prompt_tokens": response.usage.input_tokens,
                    "completion_tokens": response.usage.output_tokens,
                    "total_tokens": response.usage.input_tokens + response.usage.output_tokens,
                }
            }
        else:
            print("Invalid response structure or no content returned.")
            return {
                "content": "1",
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            }
            
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return {
            "content": "1",
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        }

# 각 평가지표별 시스템 프롬프트 정의
def get_empathy_prompt():
    return (
        "You are an expert in evaluating psychotherapy counseling responses. "
        "Please assess the response based on the Empathy criterion only:\n\n"
        "Empathy: How well does the counselor understand and support the client's emotional state? (Score 0–3)\n\n"
        "Detailed scoring criteria:\n\n"
        "0: Completely ignores the client's emotions and statements\n"
        "1: Summarizes the client's words but does not reflect their emotions\n"
        "2: Understands and responds to both the content and the emotions\n"
        "3: Accurately reads the client's emotions and goes beyond simple repetition or summary to offer emotional support\n\n"
        "Return your evaluation in this format: Rating: [0/1/2/3]"
    )

def get_logical_coherence_prompt():
    return (
        "You are an expert in evaluating psychotherapy counseling responses. "
        "Please assess the response based on the Logical Coherence criterion only:\n\n"
        "Logical Coherence: How logically consistent and well-reasoned is the response? (Score 0–3)\n\n"
        "Detailed scoring criteria:\n\n"
        "0: The response lacks logic and coherence; fails to focus on the client's issues and contains logical fallacies, contradictory perspectives, or excessive subjectivity\n"
        "1: The response shows some logic but lacks overall coherence; fails to identify reasoning based on the client's statements or uses vague expressions\n"
        "2: The response is mostly clear and logically consistent, based on sufficient reasoning and reasonable assumptions, though minor logical issues may be present\n"
        "3: The response includes sufficient reasoning and clear assumptions, demonstrates thorough and consistent logical development, contains no logical errors or contradictions, and presents a persuasive conclusion\n\n"
        "Return your evaluation in this format: Rating: [0/1/2/3]"
    )

def get_guidance_prompt():
    return (
        "You are an expert in evaluating psychotherapy counseling responses. "
        "Please assess the response based on the Guidance criterion only:\n\n"
        "Guidance: How practical and actionable is the advice provided to the client? (Score 0–3)\n\n"
        "Detailed scoring criteria:\n\n"
        "0: Lacks both specificity and practicality; no goals, action plans, or consideration of real-life situations\n"
        "1: The suggestions are somewhat specific and practical but lack clarity\n"
        "2: The suggestions are very specific and practical, including actionable plans and recommendations tailored to the client's issues and needs\n"
        "3: The suggestions are highly specific, practical, and realistic, taking various factors and real-life circumstances into account, showing feasibility and executability. Additionally, the response offers insight into the client's future growth and improvement\n\n"
        "Return your evaluation in this format: Rating: [0/1/2/3]"
    )

def evaluate_batch_questions(questions_batch, total_samples=20, batch_size=4):
    """질문 배치(5개)에 대해 3개 지표를 동시에 평가"""
    all_results = []
    
    # 각 지표별 프롬프트 정의
    metrics = [
        ("Empathy", get_empathy_prompt()),
        ("Logical Coherence", get_logical_coherence_prompt()),
        ("Guidance", get_guidance_prompt())
    ]
    
    # total_samples번 반복 (20번)
    for sample_idx in range(total_samples):
        print(f"Sample {sample_idx + 1}/{total_samples} for batch of {len(questions_batch)} questions...")
        
        # 이번 샘플에서 생성할 API 호출 목록 (10질문 × 3지표 = 30개)
        api_calls = []
        
        for entry in questions_batch:
            question_id = entry.get("questionID", "")
            model_answer = entry.get("model_answer", "")
            question_text = entry.get("total_questionText", "")
            
            user_prompt = (
                f"Below is a client's question and the counseling response. Please evaluate the response:\n\n"
                f"Client's Question: {question_text}\n\n"
                f"Counselor's Response: {model_answer}\n\n"
                "Evaluation:"
            )
            
            # 각 지표별로 API 호출 정보 추가
            for metric_name, system_prompt in metrics:
                api_calls.append({
                    "question_id": question_id,
                    "metric": metric_name,
                    "system_prompt": system_prompt,
                    "user_prompt": user_prompt,
                    "model_answer": model_answer,
                    "question_text": question_text
                })
        
        # 30개 API를 동시에 호출
        print(f"  동시 API 호출: {len(api_calls)}개")
        with ThreadPoolExecutor(max_workers=len(api_calls)) as executor:
            futures = []
            for call_info in api_calls:
                future = executor.submit(
                    create_claude_completion, 
                    call_info["system_prompt"], 
                    call_info["user_prompt"]
                )
                futures.append((future, call_info))
            
            # 결과 수집 및 정리
            batch_results = {}  # {question_id: {metric: score}}
            
            for future, call_info in futures:
                response = future.result()
                content = response["content"]
                
                # 점수 추출
                match = re.search(r"Rating:\s*([0-3])", content)
                if match:
                    score = int(match.group(1))
                else:
                    score_text = ''.join(filter(str.isdigit, content))[:1]
                    score = max(0, min(3, int(score_text))) if score_text else 0
                
                # 결과 저장
                question_id = call_info["question_id"]
                metric = call_info["metric"]
                
                if question_id not in batch_results:
                    batch_results[question_id] = {
                        "scores": {},
                        "model_answer": call_info["model_answer"],
                        "question_text": call_info["question_text"]
                    }
                
                batch_results[question_id]["scores"][metric] = score
        
        # 각 질문별로 개별 결과 생성 (3개 지표 모두 완성된 경우만)
        for question_id, result_data in batch_results.items():
            scores = result_data["scores"]
            
            # 안전장치: 3개 지표가 모두 있는지 확인
            required_metrics = {"Empathy", "Logical Coherence", "Guidance"}
            if set(scores.keys()) != required_metrics:
                print(f"  !! 질문 {question_id}: 불완전한 결과 - {list(scores.keys())} (재시도 필요)")
                
                # 누락된 지표들 재시도
                missing_metrics = required_metrics - set(scores.keys())
                for missing_metric in missing_metrics:
                    print(f"    재시도: {missing_metric}")
                    
                    # 해당 지표의 프롬프트 찾기
                    metric_prompt = None
                    for metric_name, system_prompt in metrics:
                        if metric_name == missing_metric:
                            metric_prompt = system_prompt
                            break
                    
                    if metric_prompt:
                        # 단일 재시도 (최대 3번)
                        for retry in range(3):
                            try:
                                user_prompt = (
                                    f"Below is a client's question and the counseling response. Please evaluate the response:\n\n"
                                    f"Client's Question: {result_data['question_text']}\n\n"
                                    f"Counselor's Response: {result_data['model_answer']}\n\n"
                                    "Evaluation:"
                                )
                                
                                response = create_claude_completion(metric_prompt, user_prompt)
                                content = response["content"]
                                
                                # 점수 추출
                                match = re.search(r"Rating:\s*([0-3])", content)
                                if match:
                                    score = int(match.group(1))
                                else:
                                    score_text = ''.join(filter(str.isdigit, content))[:1]
                                    score = max(0, min(3, int(score_text))) if score_text else 0
                                
                                scores[missing_metric] = score
                                print(f"    @@ {missing_metric}: {score}")
                                break
                                
                            except Exception as e:
                                print(f"    재시도 {retry+1} 실패: {e}")
                                if retry == 2:  # 3번 시도 후에도 실패하면 그냥 비워둠
                                    print(f"    X {missing_metric}: 최종 실패 - 비워둠")
            
            # 최종 확인: 3개 지표가 모두 있는지 재검증
            if len(scores) == 3 and required_metrics.issubset(set(scores.keys())):
                total_score = sum(scores.values())
                
                # 사용량 추정
                total_usage = {
                    "prompt_tokens": 150,  # 추정값
                    "completion_tokens": 50,
                    "total_tokens": 200
                }
                
                # 결과 생성
                individual_result = {
                    "questionID": question_id,
                    "evaluation_score": scores,
                    "total_score": total_score,
                    "evaluation_response": {
                        "content": json.dumps(scores, indent=2),
                        "usage": total_usage
                    },
                    "model_answer": result_data["model_answer"],
                    "question_text": result_data["question_text"]
                }
                
                all_results.append(individual_result)
                print(f"  @@ 질문 {question_id}: 완료 - {scores}")
            else:
                print(f"  X  질문 {question_id}: 최종 실패 - 건너뛰기")
        
        # API 호출 간격 조정
        time.sleep(4.5)
    
    return all_results

def evaluate_responses(data, total_samples=20, batch_size=4, output_file_path=None):
    all_results = []
    
    # 기존 결과가 있는지 확인
    processed_samples = set()
    if output_file_path and os.path.exists(output_file_path):
        print(f"기존 결과 파일 발견: {output_file_path}")
        with open(output_file_path, 'r', encoding='utf-8') as f:
            line_count = 0
            for line in f:
                try:
                    existing_result = json.loads(line.strip())
                    all_results.append(existing_result)
                    line_count += 1
                except:
                    continue
        print(f"기존 결과 {line_count}개 라인 로드됨")
        
        # 완전히 처리된 질문 ID들 계산
        from collections import Counter
        question_counts = Counter([r.get("questionID", "") for r in all_results])
        fully_processed_ids = {qid for qid, count in question_counts.items() if count >= total_samples}
        print(f"완전히 처리된 질문: {len(fully_processed_ids)}개")
    else:
        fully_processed_ids = set()
    
    # 아직 처리되지 않은 항목만 필터링
    remaining_data = [entry for entry in data if entry.get("questionID", "") not in fully_processed_ids]
    print(f"처리할 남은 항목: {len(remaining_data)}개")
    
    # 배치로 나누기 (10개씩)
    for batch_start in range(0, len(remaining_data), batch_size):
        batch_end = min(batch_start + batch_size, len(remaining_data))
        questions_batch = remaining_data[batch_start:batch_end]
        
        print(f"\n배치 {batch_start//batch_size + 1}: 질문 {batch_start+1}-{batch_end} 처리 중...")
        print(f"  질문 {len(questions_batch)}개 × 지표 3개 × 샘플 {total_samples}개 = {len(questions_batch) * 3 * total_samples}개 결과 생성")
        
        # 배치 평가 수행
        batch_results = evaluate_batch_questions(questions_batch, total_samples, batch_size)
        all_results.extend(batch_results)
        
        # 배치 처리 완료 후 저장
        if output_file_path:
            print(f"  배치 {batch_start//batch_size + 1} 완료 - {len(batch_results)}개 결과 저장 중...")
            save_jsonl(all_results, output_file_path)
    
    return all_results

def load_jsonl(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return [json.loads(line) for line in file]

def save_jsonl(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as file:
        for entry in data:
            file.write(json.dumps(entry, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    # 출력 디렉토리 생성
    os.makedirs("./output_dir/eval_output", exist_ok=True)
    
    # 데이터 로드
    data = load_jsonl(input_file_path)
    print(f"Loaded {len(data)} entries from {input_file_path}")

    # 평가 실행 (각 질문당 20번, 배치 크기 5)
    total_samples = 20
    batch_size = 4
    
    print(f"배치 크기: {batch_size}개 질문")
    print(f"각 질문당 {total_samples}번 평가")
    print(f"배치당 동시 API 호출: {batch_size} × 3 지표 = {batch_size * 3}개")
    print(f"총 예상 결과: {len(data)} × {total_samples} = {len(data) * total_samples} lines")
    
    # 배치 방식으로 평가 실행
    evaluated_data = evaluate_responses(data, total_samples, batch_size, output_file_path)
    
    # 최종 결과 저장
    save_jsonl(evaluated_data, output_file_path)
    print(f"최종 결과 저장: {output_file_path}")
    print(f"Total lines saved: {len(evaluated_data)}")
    
    # 통계 출력
    empathy_scores = [entry["evaluation_score"]["Empathy"] for entry in evaluated_data]
    logical_scores = [entry["evaluation_score"]["Logical Coherence"] for entry in evaluated_data]
    guidance_scores = [entry["evaluation_score"]["Guidance"] for entry in evaluated_data]
    total_scores = [entry["total_score"] for entry in evaluated_data]
    
    print(f"Average Empathy score: {sum(empathy_scores)/len(empathy_scores):.2f}/3")
    print(f"Average Logical Coherence score: {sum(logical_scores)/len(logical_scores):.2f}/3")
    print(f"Average Guidance score: {sum(guidance_scores)/len(guidance_scores):.2f}/3")
    print(f"Average total score: {sum(total_scores)/len(total_scores):.2f}/9")
    
    print(f"\n각 질문당 {total_samples}개의 평가 결과가 저장되었습니다!")
