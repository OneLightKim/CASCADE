import os
import json
from neo4j import GraphDatabase
from typing import List, Dict, Optional
import warnings
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder
from sklearn.metrics.pairwise import cosine_similarity
import torch

# 불필요한 경고 메시지 무시
warnings.filterwarnings("ignore")

class MentalHealthTreatmentFinder:
    """
    PrimeKG에서 다단계 탐색을 수행하는 클래스
    """
    
    def __init__(self, uri: str = "bolt://localhost:7687", user: str = "neo4j", password: str = "dkahdkah10"):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        
        # DPR 모델 (Dense Passage Retrieval)
        self.dpr_model = SentenceTransformer("ncbi/MedCPT-Query-Encoder")
        
        # Cross-encoder 모델 (Re-ranking)
        self.cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        
        # 정신건강 관련 키워드
        self.condition_keywords = {
            'anxiety': ['anxiety', 'anxiety disorder', 'generalized anxiety disorder', 'Panic disorder', 'social anxiety disorder', 'phobia', 'separation anxiety disorder'],
            'bipolar': ['bipolar disorder', 'cyclothymic disorder', 'Bipolar type I disorder', 'Bipolar type II disorder'],
            'depression': ['depression', 'Major depressive disorder', 'depressive disorder', 'Recurrent Depressive disorder', "Single episode depressive disorder", 'Dysthymic disorder', 'postpartum depression'],
            'Eating_disorder': ['eating disorder', 'anorexia', 'anorexia nervosa', 'bulimia nervosa', 'binge eating','feeding and eating disorder']
        }
        
        # 미리 계산된 evidence score 데이터 로드
        self.evidence_data = self.load_evidence_data()
        
        print(f"@@ Mental Health Treatment Finder 초기화 완료")
        print(f"@@ Evidence 데이터 로드: {len(self.evidence_data)}개 질환")

    def load_evidence_data(self) -> Dict:
        """
        미리 계산된 evidence score 데이터를 로드
        """
        evidence_file = "./src/kg2subgraph/test_label2triple_with_evidence.json"
        
        try:
            with open(evidence_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 조건별로 정리된 evidence 데이터 생성
            evidence_dict = {}
            for condition_data in data:
                if condition_data['status'] == 'success':
                    condition = condition_data['condition']
                    evidence_dict[condition] = {}
                    
                    # 각 triple의 drug name을 키로 하여 evidence score 저장
                    for triple in condition_data['all_triples']:
                        drug_name = triple['end_node']['name']
                        evidence_dict[condition][drug_name] = {
                            'evidence_score': triple.get('evidence_score', 0.0),
                            'max_phase': triple.get('max_phase', 0.0),
                            'activity_ratio': triple.get('activity_ratio', 0.0),
                            'phase_weight': triple.get('phase_weight', 0.0),
                            'total_activities': triple.get('total_activities', 0),
                            'effective_activities': triple.get('effective_activities', 0)
                        }
            
            return evidence_dict
            
        except Exception as e:
            print(f"!! Evidence 데이터 로드 실패: {e}")
            return {}

    def close(self):
        self.driver.close()

    def find_condition_start_nodes(self, condition: str) -> List[Dict]:
        if condition not in self.condition_keywords:
            print(f"X 지원하지 않는 질환: {condition}")
            return []
            
        keywords = self.condition_keywords[condition]
        print(f"@@ '{condition}' 관련 시작 노드 검색 중...")
        
        with self.driver.session(database="neo4j") as session:
            keyword_conditions = " OR ".join([
                f"toLower(n.name) CONTAINS toLower('{keyword}')" for keyword in keywords
            ])
            
            query = f"""
            MATCH (n:Node {{type: 'disease'}})
            WHERE n.name IS NOT NULL AND ({keyword_conditions})
            RETURN DISTINCT n.id AS id, n.name AS name
            LIMIT 20
            """
            
            result = session.run(query)
            nodes = [{'id': record['id'], 'name': record['name']} for record in result]
            
            if not nodes:
                print(f"X '{condition}' 관련 시작 노드를 찾을 수 없습니다.")
                return []
            
            print(f"@@ {len(nodes)}개의 잠재적 시작 노드 발견")
            return nodes

    def collect_treatment_triples(self, condition: str, start_nodes: List[Dict]) -> List[Dict]:
        if not start_nodes:
            return []
            
        print(f"🔗 {len(start_nodes)}개 시작 노드로부터 치료 관계 탐색 시작...")
        
        all_triples = []
        start_node_ids = [node['id'] for node in start_nodes]
        
        with self.driver.session(database="neo4j") as session:
            # indication과 off-label use 관계만 검색
            query = """
            MATCH (start:Node)-[r:RELATES]->(treatment:Node {type: 'drug'})
            WHERE start.id IN $start_node_ids AND r.type IN ['indication', 'off-label use']
            RETURN start.name AS start_name, r.type AS rel_type, treatment.name AS end_name, treatment.type AS end_type
            """
            result = session.run(query, start_node_ids=start_node_ids)
            for record in result:
                all_triples.append({
                    'start_node': {'name': record['start_name'], 'type': 'disease'},
                    'relation': record['rel_type'],
                    'end_node': {'name': record['end_name'], 'type': record['end_type']},
                    'priority': 1
                })
        
        # 중복 제거
        seen = set()
        unique_triples = []
        for triple in all_triples:
            key = (triple['start_node']['name'], triple['relation'], triple['end_node']['name'])
            if key not in seen:
                unique_triples.append(triple)
                seen.add(key)

        print(f"(약물) 총 {len(unique_triples)}개의 유니크한 치료 관련 Triple 수집 완료")
        
        # 미리 계산된 evidence score 정보 추가
        if condition in self.evidence_data:
            for triple in unique_triples:
                drug_name = triple['end_node']['name']
                if drug_name in self.evidence_data[condition]:
                    evidence_info = self.evidence_data[condition][drug_name]
                    triple.update(evidence_info)
                    print(f"📊 {drug_name}: Evidence score({evidence_info['evidence_score']:.3f})")
                else:
                    # Evidence 데이터가 없는 경우 기본값 설정
                    triple.update({
                        'evidence_score': 0.0,
                        'max_phase': 0.0,
                        'activity_ratio': 0.0,
                        'phase_weight': 0.0,
                        'total_activities': 0,
                        'effective_activities': 0
                    })
                    print(f"!! {drug_name}: Evidence 데이터 없음 (기본값 0.0 사용)")
        else:
            print(f"!! '{condition}' 질환의 evidence 데이터를 찾을 수 없습니다.")
            # 모든 triple에 기본값 설정
            for triple in unique_triples:
                triple.update({
                    'evidence_score': 0.0,
                    'max_phase': 0.0,
                    'activity_ratio': 0.0,
                    'phase_weight': 0.0,
                    'total_activities': 0,
                    'effective_activities': 0
                })
        
        return unique_triples

    def get_treatment_recommendation(self, condition: str, user_query: str) -> Dict:
        print(f"\n@@ '{condition.upper()}' 치료법 추천 파이프라인 시작")
        print(f"@@ 쿼리: '{user_query[:100]}...'")
        print("="*60)
        
        start_nodes = self.find_condition_start_nodes(condition)

        if not start_nodes:
            return {'status': 'no_start_nodes', 'message': f"'{condition}'에 해당하는 질병 노드를 찾지 못했습니다."}
        
        triples = self.collect_treatment_triples(condition, start_nodes)
        if not triples:
            return {'status': 'no_triples', 'message': "관련된 치료법 정보를 찾지 못했습니다."}
        
        # BM25 + DPR + Cross-encoder + Evidence support를 사용한 triple 선정
        selected_triple = self.select_best_triples_with_retrieval(user_query, triples)
        
        if not selected_triple:
            return {'status': 'no_selected_triples', 'message': "사용자 쿼리와 관련성 높은 치료법을 선별하지 못했습니다."}
        
        result = {
            'status': 'success',
            'condition': condition,
            'user_query': user_query,
            'all_triples': triples,
            'selected_triple': selected_triple,
            'total_triples_found': len(triples),
            'selected_count': 1,
            'summary': f"'{condition}' 질환에 대해 총 {len(triples)}개의 치료법 관련 triple을 찾았고, 그 중 1개를 선별했습니다."
        }
        
        print(f"@@ 결과: {result['summary']}")
        print("="*60)
        
        return result

    def convert_triples_to_sentences(self, triples: List[Dict]) -> List[str]:
        """
        Triple들을 자연어 문장으로 변환
        """
        sentences = []
        for triple in triples:
            disease = triple['start_node']['name']
            relation = triple['relation']
            treatment = triple['end_node']['name']
            
            if relation == 'indication':
                sentence = f"{treatment} is an indication for {disease}"
            elif relation == 'off-label use':
                sentence = f"{treatment} is used off-label for {disease}"
            else:
                sentence = f"{disease} {relation} {treatment}"
            
            sentences.append(sentence)
        
        return sentences

    def bm25_retrieval(self, query: str, sentences: List[str], top_k: int = 10) -> List[int]:
        """
        BM25를 사용한 sparse retrieval
        """
        # 토큰화 (간단한 공백 기반)
        tokenized_sentences = [sentence.lower().split() for sentence in sentences]
        tokenized_query = query.lower().split()
        
        # BM25 모델 생성
        bm25 = BM25Okapi(tokenized_sentences)
        
        # 검색 실행
        scores = bm25.get_scores(tokenized_query)
        
        # 상위 k개 인덱스 반환
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        return top_indices.tolist()

    def dpr_retrieval(self, query: str, sentences: List[str], top_k: int = 10) -> List[int]:
        """
        DPR을 사용한 dense retrieval
        """
        # Query와 sentences를 벡터로 변환
        query_embedding = self.dpr_model.encode([query])
        sentence_embeddings = self.dpr_model.encode(sentences)
        
        # 코사인 유사도 계산
        similarities = cosine_similarity(query_embedding, sentence_embeddings)[0]
        
        # 상위 k개 인덱스 반환
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        return top_indices.tolist()

    def cross_encoder_reranking(self, query: str, sentences: List[str], candidate_indices: List[int], top_k: int = 5) -> List[Dict]:
        """
        Cross-encoder를 사용한 re-ranking
        """
        # 후보 문장들 준비
        candidate_sentences = [sentences[i] for i in candidate_indices]
        
        # Query-sentence 쌍 생성
        query_sentence_pairs = [[query, sentence] for sentence in candidate_sentences]
        
        # Cross-encoder로 relevance score 계산
        scores = self.cross_encoder.predict(query_sentence_pairs)
        
        # 점수와 인덱스를 함께 정렬
        scored_candidates = [
            {
                'index': candidate_indices[i],
                'sentence': candidate_sentences[i],
                'relevance_score': float(scores[i])
            }
            for i in range(len(candidate_indices))
        ]
        
        # relevance score 기준으로 정렬
        scored_candidates.sort(key=lambda x: x['relevance_score'], reverse=True)
        
        return scored_candidates[:top_k]

    def select_best_triples_with_retrieval(self, user_query: str, triples: List[Dict]) -> Dict:
        """
        BM25 + DPR + Cross-encoder + 미리 계산된 Evidence score를 사용한 최적 triple 1개 선정
        """
        if not triples:
            return None
        
        print(f"🔍 Triple 선정 시작: {len(triples)}개 후보 → BM25 + DPR + Cross-encoder + Evidence → 최종 1개 선정")
        
        # 1단계: Triple을 자연어 문장으로 변환
        sentences = self.convert_triples_to_sentences(triples)
        print(f"{len(sentences)}개 문장으로 변환 완료")
        
        # 2단계: BM25 retrieval (상위 20개)
        bm25_candidates = self.bm25_retrieval(user_query, sentences, top_k=min(20, len(sentences)))
        print(f"BM25 후보: {len(bm25_candidates)}개")
        
        # 3단계: DPR retrieval (상위 20개)
        dpr_candidates = self.dpr_retrieval(user_query, sentences, top_k=min(20, len(sentences)))
        print(f"DPR 후보: {len(dpr_candidates)}개")
        
        # 4단계: 두 방법의 후보를 합치고 중복 제거
        combined_candidates = list(set(bm25_candidates + dpr_candidates))
        print(f"통합 후보: {len(combined_candidates)}개")
        
        # 5단계: Cross-encoder re-ranking (상위 5개만 계산)
        reranked_results = self.cross_encoder_reranking(
            user_query, sentences, combined_candidates, top_k=5
        )
        print(f"@@ Re-ranking 완료: 상위 5개 후보")
        
        # 6단계: 미리 계산된 Evidence score와 Relevance score 합산하여 최종 점수 계산
        print(f"@@ 최종 점수 계산: Relevance Score + Evidence Score")
        
        for result in reranked_results:
            triple_idx = result['index']
            triple = triples[triple_idx]
            
            # 미리 계산된 evidence score 가져오기
            evidence_score = triple.get('evidence_score', 0.0)
            
            # 최종 점수 = relevance_score + evidence_score
            final_score = result['relevance_score'] + evidence_score
            
            result['evidence_score'] = evidence_score
            result['final_score'] = final_score
            
            print(f"  📊 {triple['end_node']['name']}: "
                  f"Relevance({result['relevance_score']:.3f}) + "
                  f"Evidence({evidence_score:.3f}) = "
                  f"Final({final_score:.3f})")
        
        # 7단계: 최종 점수로 재정렬
        reranked_results.sort(key=lambda x: x['final_score'], reverse=True)
        
        # 8단계: 최종 1개 선정
        if not reranked_results:
            return None
            
        best_result = reranked_results[0]
        
        # 원본 triple과 선정 기준 정보를 함께 반환
        final_triple = triples[best_result['index']].copy()
        final_triple['relevance_score'] = best_result['relevance_score']
        final_triple['final_score'] = best_result['final_score']
        final_triple['natural_sentence'] = best_result['sentence']
        
        # 선정 기준 정보 추가
        selection_criteria = {
            'method': 'BM25 + DPR + Cross-encoder + Pre-calculated Evidence Score',
            'total_candidates': len(triples),
            'bm25_candidates': len(bm25_candidates),
            'dpr_candidates': len(dpr_candidates),
            'combined_candidates': len(combined_candidates),
            'cross_encoder_score': best_result['relevance_score'],
            'evidence_score': best_result['evidence_score'],
            'final_combined_score': best_result['final_score'],
            'ranking_position': 1,
            'selection_reason': f"최종 점수 ({best_result['final_score']:.4f}) = Relevance({best_result['relevance_score']:.3f}) + Evidence({best_result['evidence_score']:.3f})"
        }
        
        final_triple['selection_criteria'] = selection_criteria
        
        print(f"@@ 최종 선정: '{final_triple['end_node']['name']}' (최종점수: {best_result['final_score']:.4f})")
        print(f"@@ 선정 기준: {selection_criteria['selection_reason']}")
        print(f"@@ Evidence 상세: Phase({final_triple.get('max_phase', 0):.1f}), "
              f"Activities({final_triple.get('effective_activities', 0)}/{final_triple.get('total_activities', 0)}), "
              f"Activity_ratio({final_triple.get('activity_ratio', 0):.3f})")
        
        return final_triple


# 파이프라인에서 호출할 수 있는 함수 인터페이스 추가
def get_treatment_triple_from_condition(condition: str, user_query: str, 
                                      neo4j_uri: str = "bolt://localhost:7687", 
                                      neo4j_user: str = "neo4j", 
                                      neo4j_password: str = "dkahdkah10") -> Dict:
    """
    파이프라인에서 호출할 수 있는 함수
    특정 질환과 사용자 쿼리에 대한 치료법 Triple을 반환
    """
    finder = MentalHealthTreatmentFinder(uri=neo4j_uri, user=neo4j_user, password=neo4j_password)
    
    try:
        result = finder.get_treatment_recommendation(condition, user_query)
        
        # 결과 포맷을 pipeline에 맞게 조정
        if result['status'] == 'success':
            selected_triple = result['selected_triple']
            
            # summary 정보 추가
            result['summary'] = {
                'selected_treatment': selected_triple['end_node']['name'],
                'selected_start_node': selected_triple['start_node']['name'],
                'relation': selected_triple['relation'],
                'final_score': selected_triple.get('final_score', 0.0),
                'relevance_score': selected_triple.get('relevance_score', 0.0),
                'evidence_score': selected_triple.get('evidence_score', 0.0)
            }
        
        return result
        
    except Exception as e:
        return {
            'status': 'error',
            'message': f"Error processing {condition}: {str(e)}",
            'condition': condition,
            'user_query': user_query
        }
    finally:
        finder.close()


def main():
    NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
    NEO4J_USER = os.environ.get("NEO4J_USER", "neo4j")
    NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "dkahdkah10")

    finder = MentalHealthTreatmentFinder(uri=NEO4J_URI, user=NEO4J_USER, password=NEO4J_PASSWORD)

    try:
        test_cases = {
            'depression': "I'm suffering from major depression and feel hopeless. What treatment options are available?",
            'anxiety': "I have been experiencing severe anxiety and panic attacks lately. What treatments would be most effective?",
            'bipolar': "I have bipolar disorder with manic episodes. What medication can help stabilize my mood?",
            'Eating_disorder': "I have anorexia nervosa and can't stop restricting food. What treatments can help?"
        }
        
        all_results = []
        
        # 각 질환별로 치료법 추천
        for condition, query in test_cases.items():
            print(f"\n{'='*80}")
            print(f"처리 중: {condition}")
            print(f"{'='*80}")
            
            result = finder.get_treatment_recommendation(condition, query)
            all_results.append(result)
        
        # 전체 결과 저장
        output_dir = "./output_dir"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "test_final_recommendations.json")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        
        # 전체 통계 출력
        print(f"\n{'='*80}")
        print("@@ 전체 결과 요약:")
        print(f"{'='*80}")
        
        for i, result in enumerate(all_results):
            condition = list(test_cases.keys())[i]
            if result['status'] == 'success':
                selected_triple = result['selected_triple']
                treatment_name = selected_triple['end_node']['name']
                final_score = selected_triple['final_score']
                relevance_score = selected_triple['relevance_score']
                evidence_score = selected_triple.get('evidence_score', 0.0)
                
                print(f"** {condition:15}: {treatment_name} "
                      f"(Final: {final_score:.3f} = Rel: {relevance_score:.3f} + Evi: {evidence_score:.3f})")
            else:
                print(f"X {condition:15}: 추천 실패")
        
        print(f"\n@@ 최종 추천 결과가 {output_path}에 저장되었습니다.")

    except Exception as e:
        print(f"X 메인 실행 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
    finally:
        finder.close()
        print("\n** 파이프라인 종료.")


if __name__ == "__main__":
    main()
