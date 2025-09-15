import os
import json
import argparse
from tqdm import tqdm
from attrdict import AttrDict
import torch
from torch.nn.functional import softmax
from transformers import AutoTokenizer, AutoModel, AutoConfig
from transformers.modeling_utils import PreTrainedModel
import torch.nn as nn
from openai import OpenAI

# KG Triple 추출 모듈
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.kg2subgraph.label2triple import get_treatment_triple_from_condition
# RAG 모듈
from src.RAG.Run_RAG import get_treatment_context_from_triple

# strategy_info.json 파일 경로 설정 (현재 디렉토리 기준)
current_dir = "./src/"
strategy_file_path = os.path.join(current_dir, "strategy_info.json")

try:
    with open(strategy_file_path, "r", encoding="utf-8") as f:
        strategy_info = json.load(f)
    print(f"** Strategy info loaded from: {strategy_file_path}")
except Exception as e:
    print(f"X Error loading strategy_info.json: {e}")
    # 기본값 설정
    strategy_info = {
        "depression": "Depression counseling strategy not available.",
        "anxiety": "Anxiety counseling strategy not available.",
        "bipolar": "Bipolar counseling strategy not available.",
        "Eating_disorder": "Eating disorder counseling strategy not available."
    }



# --------------------------------------------------------------------------------
# 1. ChatGPT API 호출을 위한 함수 (첫 번째 스크립트에서 가져옴)
# --------------------------------------------------------------------------------
def create_chat_completion(client, system_prompt, user_prompt, model="gpt-4o", temperature=0.8, max_tokens=650):
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
# 2. RoBERTa 분류 모델 관련 함수 (기존과 동일)
# --------------------------------------------------------------------------------

# 질환명 매핑 (RoBERTa 분류 결과 → KG 검색용)
CONDITION_MAPPING = {
    'anxiety': 'anxiety',
    'bipolar': 'bipolar',  # Bipolar Disorder, manic depression으로 검색
    'depression': 'depression',
    'Eating_disorder': 'Eating_disorder',  # 대문자 E 버전도 대비
    'control': 'control'  # 일반 상담
}

class MentalROBERTA_SENTIMENT_CLASSIFIER(PreTrainedModel):
    def __init__(self, config):
        super(MentalROBERTA_SENTIMENT_CLASSIFIER, self).__init__(config)
        self.roberta = AutoModel.from_pretrained(config._name_or_path)
        self.hidden_size = config.hidden_size
        self.num_labels = config.num_labels
        self.relu = nn.ReLU()
        self.linear1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.linear2 = nn.Linear(self.hidden_size, self.num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        cls_vector = outputs.last_hidden_state[:, 0, :]
        linear_output = self.linear1(cls_vector)
        cls_output = self.linear2(linear_output)
        return cls_output

def load_roberta_model(model_dir):
    try:
        roberta_config = AutoConfig.from_pretrained(model_dir)
        setattr(roberta_config, "num_labels", 5)
        setattr(roberta_config, "hidden_size", 768)
        model = MentalROBERTA_SENTIMENT_CLASSIFIER.from_pretrained(
            model_dir, config=roberta_config
        )
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        model.eval()
        return model, tokenizer, device
    except Exception as e:
        raise Exception(f"RoBERTa 모델 로딩 중 오류 발생: {str(e)}")

def predict_label(model, tokenizer, device, sentence):
    try:
        inputs = tokenizer(sentence, return_tensors="pt", truncation=True, padding=True, max_length=128)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(inputs['input_ids'], inputs['attention_mask'])
            probs = softmax(outputs, dim=-1)
            pred = torch.argmax(probs, dim=-1).item()
        return pred
    except Exception as e:
        raise Exception(f"레이블 예측 중 오류 발생: {str(e)}")

# --------------------------------------------------------------------------------
# 3. 메인 파이프라인 함수 (LLaMA 호출 부분을 ChatGPT 호출로 변경)
# --------------------------------------------------------------------------------
def main(cli_args):
    try:
        args = AttrDict(vars(cli_args))
        
        # OpenAI 클라이언트 초기화
        client = OpenAI(api_key=args.api_key)
        
        # RoBERTa 모델 로딩 (기존과 동일)
        roberta_model, roberta_tokenizer, device = load_roberta_model("./src/MentalRoBERTa/0421")

        # 레이블 매핑 로드
        # with open("./src/MentalRoBERTa/0421/label2idx.json", 'r', encoding='utf-8') as f:
        #     label2idx = json.load(f)
        # idx2label = {str(v): k for k, v in label2idx.items()}
        # 라벨 매핑 로드 (올바른 버전)
        with open("./src/MentalRoBERTa/0421/label2idx.json", 'r', encoding='utf-8') as f:
            label_mapping = json.load(f)
        label2idx = label_mapping["label2idx"]  # 중첩된 구조에서 추출
        idx2label = {str(v): k for k, v in label2idx.items()}

        # PsyQA 번역 데이터 로드 (처음 1000개만)
        print("Loading translated PsyQA dataset...")
        with open(args.data_file, 'r', encoding='utf-8') as f:
            full_dataset = json.load(f)
            
        # 400개 샘플 파일을 전체 사용
        dataset = full_dataset
        
        print(f"전체 데이터셋 크기: {len(full_dataset)}")
        print(f"처리할 데이터 크기: {len(dataset)} (400개 샘플)")

        # 출력 디렉토리 생성
        os.makedirs(args.output_dir, exist_ok=True)
        # 출력 파일을 하나로 통일하여 간결하게 만듦
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
                print(f"** 이미 처리된 샘플: {len(processed_ids)}개")
                print(f"인덱스 {start_idx}부터 재시작합니다.")
                
            except Exception as e:
                print(f"!! 기존 파일 읽기 오류: {e}")
                print("처음부터 다시 시작합니다.")
                processed_ids = set()
                start_idx = 0

        # 파일 모드 결정 (append 또는 write)
        file_mode = 'a' if start_idx > 0 else 'w'
        
        with open(output_path, file_mode, encoding='utf-8') as of:
            # 진행률 표시를 위한 tqdm 설정
            remaining_dataset = dataset[start_idx:] if start_idx > 0 else dataset
            progress_desc = f"Processing samples ({start_idx+1}-{len(dataset)}) with real-time RAG" if start_idx > 0 else f"Processing {len(dataset)} samples with real-time RAG"
            
            for relative_idx, data in enumerate(tqdm(remaining_dataset, desc=progress_desc, initial=start_idx, total=len(dataset))):
                try:
                    # 실제 인덱스 계산 (전체 데이터셋에서의 위치)
                    actual_idx = start_idx + relative_idx
                    
                    # PsyQA 데이터 구조에 맞게 필드 추출
                    question_id = data.get("idx", "")  # 새로 추가된 idx 사용
                    
                    # 이미 처리된 ID인지 확인 (중복 방지)
                    if question_id in processed_ids:
                        print(f"⏭ 이미 처리된 샘플 건너뜀: {question_id}")
                        continue
                    
                    question_title = data.get("question", "")  # 번역된 질문
                    question_desc = data.get("description", "")  # 번역된 설명
                    answers = data.get("answers", [])
                    answer_text = answers[0].get("answer_text", "") if answers else ""

                    # 질문과 설명을 합쳐서 전체 텍스트 생성
                    post = f"{question_title}. {question_desc}"

                    # 1. RoBERTa로 레이블 예측 (기존과 동일)
                    pred_label_idx = predict_label(roberta_model, roberta_tokenizer, device, post)
                    pred_label = idx2label.get(str(pred_label_idx), "unknown")
                    # strategy = strategy_info.get(pred_label, "No strategy available.")

                    # 2. 프롬프트 생성 (시스템 프롬프트와 사용자 프롬프트 분리)
                    if pred_label == "control":
                        system_prompt = (
                            "You are a helpful and insightful general counselor. Your goal is to provide a thoughtful "
                            "perspective and practical advice for everyday life concerns and general questions. "
                            "Your tone should be supportive and encouraging, but not clinical."
                            "Please provide a comprehensive, cohesive, and empathetic response of approximately 650 tokens that flows naturally, avoids list formats, and maintains an emotionally supportive tone throughout."
                        )
                        user_prompt = (
                            f"The user's counseling content is as follows: {post}\n\n"
                            f"Based on my analysis, the category of this concern is 'General/Control', "
                            "As an advisor, an appropriate approach to support the user includes the following principles: The foundation is active and empathetic listening, which helps the user feel heard and understood. It is important to validate their feelings without judgment, creating a safe space for them to express themselves. Encouraging gentle self-reflection can help them gain clarity on their own. Finally, offering encouragement to consider small, positive steps forward can be empowering, without being prescriptive.\n\n"
                            "Please generate a compassionate, conversational, and well-structured response that reflects the user's counseling content with empathy and emotional support. "
                            "Naturally integrate the predicted mental disorder, the recommended treatment strategy, and the medication information, ensuring the response flows smoothly in a natural, non-list format, with a warm and encouraging tone."
                        )
                    else: # 정신 질환 라벨을 위한 전문 상담 프롬프트 + KG Triple & RAG 콘텐츠
                        # 3. KG Triple 추출
                        print(f"🔍 [{actual_idx+1}/{len(dataset)}] {pred_label} 질환에 대한 치료법 Triple 검색 중...")
                        mapped_condition = CONDITION_MAPPING.get(pred_label, pred_label)
                        triple_result = get_treatment_triple_from_condition(
                            condition=mapped_condition,
                            user_query=post,
                            neo4j_uri="bolt://localhost:7687",
                            neo4j_user="neo4j", 
                            neo4j_password="dkahdkah10"
                        )
                        
                        # 4. 질환별 치료 전략 가져오기
                        counseling_strategy = strategy_info.get(pred_label, strategy_info.get("depression", ""))  # 기본값으로 depression 전략 사용
                        
                        # 5. RAG 콘텐츠 생성 (실시간 처리)
                        print(f"📚 [{actual_idx+1}/{len(dataset)}] 치료법 관련 최신 논문 검색 및 콘텐츠 생성 중...")
                        
                        # RAG 처리를 통해 논문 정보와 상세 결과 모두 얻기
                        from src.RAG.Run_RAG import EnhancedRAGSystem
                        
                        # RAG 시스템으로 상세 결과 얻기
                        rag_system = EnhancedRAGSystem(margin_delta=0.03)
                        detailed_rag_result = rag_system.process_single_triple(
                            disease=triple_result['selected_triple']['start_node']['name'],
                            relation=triple_result['selected_triple']['relation'],
                            drug=triple_result['selected_triple']['end_node']['name'],
                            user_query=post,
                            condition=mapped_condition
                        )
                        
                        # 간단한 drug_treatment_info 생성 (GPT 프롬프트용)
                        if detailed_rag_result["success"]:
                            paper = detailed_rag_result["best_paper"]
                            drug_treatment_info = (
                                f"**Medication: {triple_result['selected_triple']['end_node']['name']}**\n"
                                f"Title: {paper['title']}\n"
                                f"Abstract: {paper['abstract']}"
                            )
                        else:
                            drug_treatment_info = "Professional medication evaluation and treatment should be considered as part of comprehensive care."
                        
                        system_prompt = (
                            "You are a compassionate and empathetic mental health counseling expert. "
                            "Carefully analyze the user's counseling content and respond with warmth, understanding, and non-judgment."
                            "Your response should be supportive and comforting while still being evidence-based and tailored to the user's predicted disorder, recommended strategy, and medication information."
                            "Please provide a comprehensive, cohesive, and empathetic response of approximately 650 tokens that flows naturally, avoids list formats, and maintains an emotionally supportive tone throughout."

                        )
                        user_prompt = (
                            f"The user's counseling content is: {post}\n"
                            f"The predicted user's mental disorder information is : {pred_label}\n"
                            f"the Recommended treatment strategy for this condition is: {counseling_strategy}\n"
                            f"And the Relevant medication treatment information is: {drug_treatment_info}\n"
                            "Please generate a compassionate, conversational, and well-structured response that reflects the user's counseling content with empathy and emotional support. "
                            "Naturally integrate the predicted mental disorder, the recommended treatment strategy, and the medication information, ensuring the response flows smoothly in a natural, non-list format, with a warm and encouraging tone."
                        )

                            # "Please generate a response that addresses the user's counseling content while effectively incorporating the mental disorder information, treatment strategy, and medication details."
                        # "Please generate a response that addresses the user's counseling content while effectively incorporating the mental disorder information, treatment strategy, and medication details."
                            # "Please generate a response that reflects the user's counseling content and the predicted mental disorder, incorporating the above specialized strategies and evidence-based medication information. "
                            # "The response should be empathetic, supportive, and practical, while being specific to their diagnosed condition."                        
                            # "response aimed at treating the user's mental disorder by utilizing the user's counseling content, mental disorder information, and the corresponding treatment strategy and Medication Treatment Information: "
                    # 3. ChatGPT API 호출로 텍스트 생성
                    generated = create_chat_completion(client, system_prompt, user_prompt, model="gpt-4o", max_tokens=650)

                    # 5. 결과 저장
                    # JSON 저장을 위해 줄바꿈 문자를 적절히 처리
                    result = {
                        "questionID": question_id,
                        "total_questionText": post.replace('\n', ' ').replace('\r', ' '),  # 줄바꿈 제거
                        "model_answer": generated.replace('\n', ' ').replace('\r', ' '),   # 줄바꿈 제거
                        "answerText": answer_text.replace('\n', ' ').replace('\r', ' '),   # 줄바꿈 제거
                        "predicted_label": pred_label
                    }
                    
                    # 정신질환 분류의 경우 Triple 및 RAG 정보 추가
                    if pred_label != "control" and 'triple_result' in locals() and triple_result['status'] == 'success':
                        result["kg_triple_info"] = {
                            "condition": triple_result['condition'],
                            "selected_treatment": triple_result['summary']['selected_treatment'],
                            "selected_start_node": triple_result['summary']['selected_start_node'],
                            "relation": triple_result['summary']['relation']
                        }
                        
                        # RAG 논문 상세 정보 추가
                        if 'detailed_rag_result' in locals() and detailed_rag_result["success"]:
                            best_paper = detailed_rag_result["best_paper"]
                            result["rag_paper_info"] = {
                                "pmid": best_paper.get("pmid", ""),
                                "title": best_paper.get("title", ""),
                                "abstract": best_paper.get("abstract", ""),
                                "journal": best_paper.get("journal", ""),
                                "year": best_paper.get("year", ""),
                                "mesh_terms": best_paper.get("mesh_terms", []),
                                "publication_types": best_paper.get("publication_types", []),
                                "clinical_score": best_paper.get("clinical_score", 0.0)
                            }
                            
                            # RAG 처리 과정 정보 추가
                            result["rag_process_info"] = {
                                "selection_method": detailed_rag_result["selection_info"]["selection_method"],
                                "pubmed_query": detailed_rag_result["selection_info"]["pubmed_query"],
                                "pipeline_stats": detailed_rag_result["pipeline_stats"]
                            }
                        else:
                            result["rag_paper_info"] = None
                            result["rag_process_info"] = {
                                "selection_method": "failed",
                                "reason": detailed_rag_result.get("reason", "unknown") if 'detailed_rag_result' in locals() else "no_rag_attempted"
                            }
                    of.write(json.dumps(result, ensure_ascii=False) + "\n")
                    # 매 샘플마다 파일 플러시 (진행 상황 보존)
                    of.flush()

                    # GPU 메모리 정리 (RoBERTa 사용 후)
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                except Exception as e:
                    print(f"X Error processing data for questionID {question_id}: {str(e)}")
                    # 에러가 발생해도 파일을 즉시 플러시하여 진행 상황 보존
                    of.flush()
                    continue

        print(f"Results successfully saved to {output_path}")

    except Exception as e:
        print(f"An error occurred in the main execution: {str(e)}")
        raise

if __name__ == "__main__":
    cli_parser = argparse.ArgumentParser(description="Run inference using RoBERTa for classification and ChatGPT for generation.")
    
    # 필수 인자
    cli_parser.add_argument("--api_key", type=str, default="", help="Your OpenAI API key.")
    
    # 파일 경로 인자
    # cli_parser.add_argument("--data_file", type=str, default="./src/MentalRoBERTa/0421/v2_finaldf123_inference_sample.jsonl", help="Path to the input data file.")
    cli_parser.add_argument("--data_file", type=str, default="data/main/400_psyqa_translated.json", help="Path to the 400-sample PsyQA data file.")
    cli_parser.add_argument("--output_dir", type=str, default="./output_dir/pipeline", help="Directory to save the output file.")
    cli_parser.add_argument("--output_file", type=str, default="gpt35_psyqa_inference_LSKR.jsonl", help="Name of the output file.")
    
    args = cli_parser.parse_args()
    
    # API 키를 환경 변수로 설정하는 것이 더 안전하지만, 인자로 직접 전달
    # os.environ["OPENAI_API_KEY"] = args.api_key
    
    main(args)