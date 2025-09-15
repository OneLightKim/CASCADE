import json
import statistics

def load_jsonl(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return [json.loads(line) for line in file]

def calculate_average_scores(eval_results):
    empathy_scores = []
    logical_coherence_scores = []
    guidance_scores = []
    
    for entry in eval_results:
        eval_score = entry.get("evaluation_score", {})
        
        # 각 지표별 점수 수집
        empathy_scores.append(eval_score.get("Empathy", 0))
        logical_coherence_scores.append(eval_score.get("Logical Coherence", 0))
        guidance_scores.append(eval_score.get("Guidance", 0))
    
    # 평균 계산
    avg_empathy = statistics.mean(empathy_scores) if empathy_scores else 0
    avg_logical_coherence = statistics.mean(logical_coherence_scores) if logical_coherence_scores else 0
    avg_guidance = statistics.mean(guidance_scores) if guidance_scores else 0
    
    return {
        "Empathy": round(avg_empathy, 2),
        "Logical Coherence": round(avg_logical_coherence, 2),
        "Guidance": round(avg_guidance, 2)
    }

if __name__ == "__main__":
    # LSKR 평가 결과 파일 경로
    input_file_path = "./output_dir/eval_output/ML_LSK_psyqa_inference.jsonl"
    output_file_path = "./output_dir/eval_output/average_score_ML_LSK_psyqa_inference.jsonl"
    
    try:
        # 평가 결과 로드
        eval_results = load_jsonl(input_file_path)
        
        # 평균 점수 계산
        average_scores = calculate_average_scores(eval_results)
        
        # 결과 출력
        print(average_scores)
        
        # JSON 파일로 저장
        with open(output_file_path, 'w', encoding='utf-8') as f:
            json.dump(average_scores, f, ensure_ascii=False, indent=2)
        
        print(f"결과가 {output_file_path}에 저장되었습니다.")
        
    except FileNotFoundError:
        print(f"파일을 찾을 수 없습니다: {input_file_path}")
    except Exception as e:
        print(f"오류가 발생했습니다: {e}")
