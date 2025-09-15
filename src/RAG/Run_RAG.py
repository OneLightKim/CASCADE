# -*- coding: utf-8 -*-
"""
Enhanced RAG System for Medical Triple → PubMed Paper Retrieval
새로운 플로우:
1. PubMed E-utilities로 후보 수집 (Best Match 활용)
2. MedCPT Dense 검색 (의미 매칭)
3. PubMedBERT 크로스-인코더 리랭킹
4. 마진 규칙(Margin rule) 기반 최종 선택
"""
import os
import sys
import json
import time
import requests
import xml.etree.ElementTree as ET
from typing import List, Dict, Tuple, Any, Optional
from datetime import datetime
import numpy as np

# -----------------------------
# Utils
# -----------------------------
def _safe_lower(s):
    try:
        return (s or "").lower()
    except:
        return ""

def _dedup(xs: List[str]) -> List[str]:
    seen = set()
    out = []
    for x in xs:
        x = x.strip()
        if x and x not in seen:
            seen.add(x)
            out.append(x)
    return out

# -----------------------------
# Triple Linearization
# -----------------------------
class TripleLinearizer:
    """Convert knowledge graph triples to natural language sentences"""
    
    def linearize_triple(self, disease: str, relation: str, drug: str) -> str:
        """Convert triple to natural language sentence based on relation type"""
        if relation.lower() == "indication":
            return f"{drug} is an indication for {disease}"
        elif "off-label" in relation.lower():
            return f"{drug} is used off-label for {disease}"
        else:
            # Fallback for other relations
            return f"{drug} {relation} {disease}"
    
    def create_search_queries(self, disease: str, relation: str, drug: str, user_query: str = "") -> List[str]:
        """Create multiple search query variations"""
        base_sentence = self.linearize_triple(disease, relation, drug)
        
        queries = [
            base_sentence,
            f"{drug} treatment for {disease}",
            f"{drug} therapy {disease}",
            user_query if user_query.strip() else f"treatment of {disease} with {drug}",
        ]
        
        return [q for q in queries if q.strip()]

# -----------------------------
# Medical Term Normalizer
# -----------------------------
class MedicalNormalizer:
    """Normalize medical terms using MeSH, RxNorm APIs"""
    
    def __init__(self):
        # Drug similarity mapping for common issues
        self.drug_similarity_map = {
            'clortermine': ['clomipramine'],
            'clortermin': ['clomipramine'],
            'clortermina': ['clomipramine'],
        }
    
    def normalize_drug_name(self, drug_name: str) -> str:
        """Normalize drug name using similarity mapping"""
        drug_lower = drug_name.lower().strip()
        if drug_lower in self.drug_similarity_map:
            alternatives = self.drug_similarity_map[drug_lower]
            print(f"@@ Drug name normalized: '{drug_name}' → '{alternatives[0]}'")
            return alternatives[0]
        return drug_name
    
    def get_mesh_terms(self, disease: str) -> List[str]:
        """Get MeSH terms for disease"""
        try:
            # MeSH descriptor lookup
            r = requests.get(
                "https://id.nlm.nih.gov/mesh/lookup/descriptor",
                params={"label": disease, "match": "exact"},
                timeout=15,
            )
            items = r.json()
            if not items:
                return [disease]
            
            # Get detailed MeSH record
            ui = items[0]["resource"].split("/")[-1]
            rec = requests.get(
                "https://id.nlm.nih.gov/mesh/record",
                params={"descriptor": ui},
                timeout=15,
            ).json()
            
            terms = [disease]
            if rec.get("descriptorName"):
                terms.append(rec["descriptorName"]["name"])
            
            # Add concept terms
            for concept in rec.get("conceptList", []):
                for term in concept.get("termList", []):
                    if term.get("name"):
                        terms.append(term["name"])
            
            return _dedup(terms)
        except Exception:
            return [disease]
    
    def get_drug_synonyms(self, drug: str) -> List[str]:
        """Get drug synonyms from RxNorm"""
        normalized_drug = self.normalize_drug_name(drug)
        all_names = [drug] if normalized_drug == drug else [drug, normalized_drug]
        
        # RxNorm lookup
        for drug_name in [normalized_drug]:
            try:
                # Get RxCUI
                r = requests.get(
                    "https://rxnav.nlm.nih.gov/REST/rxcui.json",
                    params={"name": drug_name},
                    timeout=15,
                ).json()
                rxcui = r.get("idGroup", {}).get("rxnormId", [])
                if rxcui:
                    # Get synonyms
                    r2 = requests.get(
                        "https://rxnav.nlm.nih.gov/REST/allProperties.json",
                        params={"rxcui": rxcui[0], "prop": "all"},
                        timeout=15,
                    ).json()
                    props = r2.get("propConceptGroup", {}).get("propConcept", []) or []
                    for p in props:
                        if p.get("propValue"):
                            all_names.append(p["propValue"])
            except Exception:
                pass
        
        return _dedup(all_names)

# -----------------------------
# PubMed Client with Best Match
# -----------------------------
class PubMedClient:
    """PubMed search client using Best Match ranking"""
    
    def create_pubmed_query(self, disease_terms: List[str], drug_terms: List[str]) -> str:
        """Create comprehensive PubMed query with clinical focus"""
        disease_mesh = " OR ".join([f'"{d}"[MeSH Terms]' for d in disease_terms])
        disease_tiab = " OR ".join([f'"{d}"[tiab]' for d in disease_terms])
        drug_substance = " OR ".join([f'"{g}"[Substance Name]' for g in drug_terms])
        drug_tiab = " OR ".join([f'"{g}"[tiab]' for g in drug_terms])
        
        disease_block = f"({disease_mesh} OR {disease_tiab})"
        drug_block = f"({drug_substance} OR {drug_tiab})"
        
        # 치료/임상 관련 키워드 추가
        clinical_terms = "(treatment[tiab] OR therapy[tiab] OR therapeutic[tiab] OR clinical[tiab] OR efficacy[tiab] OR effectiveness[tiab])"
        
        # 인간 대상 연구 우선 필터
        human_filter = "(humans[MeSH Terms] OR human[tiab] OR patient[tiab] OR patients[tiab])"
        
        return f"{disease_block} AND {drug_block} AND {clinical_terms} AND {human_filter}"
    
    def create_fallback_query(self, disease_terms: List[str], drug_terms: List[str]) -> str:
        """Create fallback query without strict filters if primary query fails"""
        disease_mesh = " OR ".join([f'"{d}"[MeSH Terms]' for d in disease_terms])
        disease_tiab = " OR ".join([f'"{d}"[tiab]' for d in disease_terms])
        drug_substance = " OR ".join([f'"{g}"[Substance Name]' for g in drug_terms])
        drug_tiab = " OR ".join([f'"{g}"[tiab]' for g in drug_terms])
        
        disease_block = f"({disease_mesh} OR {disease_tiab})"
        drug_block = f"({drug_substance} OR {drug_tiab})"
        
        return f"{disease_block} AND {drug_block}"
    
    def search_pubmed_best_match(self, query: str, max_results: int = 200) -> List[str]:
        """Search PubMed using Best Match (기본 정렬)"""
        try:
            r = requests.get(
                "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
                params={
                    "db": "pubmed", 
                    "term": query, 
                    "retmode": "json", 
                    "retmax": max_results, 
                    "sort": "relevance"  # Best Match 정렬
                },
                timeout=30,
            ).json()
            return r.get("esearchresult", {}).get("idlist", [])
        except Exception as e:
            print(f"!! PubMed search error: {e}")
            return []
    
    def fetch_articles_with_clinical_filter(self, pmids: List[str]) -> List[Dict]:
        """Fetch article details with clinical relevance filtering"""
        if not pmids:
            return []
        
        try:
            r = requests.get(
                "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi",
                params={"db": "pubmed", "id": ",".join(pmids), "retmode": "xml"},
                timeout=60,
            )
            
            articles = []
            root = ET.fromstring(r.text)
            
            for art in root.findall(".//PubmedArticle"):
                try:
                    pmid = art.findtext(".//PMID") or ""
                    title = art.findtext(".//ArticleTitle") or ""
                    
                    # Get abstract
                    abstract_parts = []
                    for abs_text in art.findall(".//AbstractText"):
                        if abs_text.text:
                            abstract_parts.append(abs_text.text)
                    abstract = " ".join(abstract_parts)
                    
                    # Get metadata
                    journal = art.findtext(".//Journal/Title") or ""
                    year = art.findtext(".//PubDate/Year") or "0000"
                    
                    # Get MeSH terms
                    mesh_terms = []
                    for mh in art.findall(".//MeshHeading"):
                        desc = mh.findtext("DescriptorName")
                        if desc:
                            mesh_terms.append(desc)
                    
                    # Get publication types
                    pub_types = []
                    for pt in art.findall(".//PublicationType"):
                        if pt.text:
                            pub_types.append(pt.text)
                    
                    # 임상 관련성 스코어 계산
                    clinical_score = self._calculate_clinical_relevance(title, abstract, mesh_terms, pub_types)
                    
                    articles.append({
                        "pmid": pmid,
                        "title": title,
                        "abstract": abstract,
                        "journal": journal,
                        "year": year,
                        "mesh_terms": mesh_terms,
                        "publication_types": pub_types,
                        "full_text": f"{title} {abstract}",
                        "clinical_score": clinical_score
                    })
                    
                except Exception as e:
                    print(f"!! Error parsing article: {e}")
                    continue
            
            # 임상 관련성이 높은 순으로 정렬
            articles.sort(key=lambda x: x["clinical_score"], reverse=True)
            return articles
            
        except Exception as e:
            print(f"!! Error fetching articles: {e}")
            return []
    
    def _calculate_clinical_relevance(self, title: str, abstract: str, mesh_terms: List[str], pub_types: List[str]) -> float:
        """Calculate clinical relevance score"""
        score = 0.0
        
        # 텍스트를 소문자로 변환
        text = f"{title} {abstract}".lower()
        mesh_lower = [term.lower() for term in mesh_terms]
        pub_lower = [pt.lower() for pt in pub_types]
        
        # 1. 인간 대상 연구 키워드 (+2.0)
        human_keywords = ["human", "humans", "patient", "patients", "clinical", "trial", "study"]
        for keyword in human_keywords:
            if keyword in text or any(keyword in mesh for mesh in mesh_lower):
                score += 2.0
                break
        
        # 2. 치료 관련 키워드 (+1.5)
        treatment_keywords = ["treatment", "therapy", "therapeutic", "efficacy", "effectiveness", "intervention"]
        for keyword in treatment_keywords:
            if keyword in text:
                score += 1.5
                break
        
        # 3. 임상시험 타입 (+3.0)
        clinical_pub_types = ["clinical trial", "randomized controlled trial", "controlled clinical trial"]
        for pub_type in clinical_pub_types:
            if any(pub_type in pt for pt in pub_lower):
                score += 3.0
                break
        
        # 4. 리뷰/메타분석 (+2.5)
        review_types = ["systematic review", "meta-analysis", "review"]
        for review_type in review_types:
            if any(review_type in pt for pt in pub_lower):
                score += 2.5
                break
        
        # 5. 동물/기초연구 페널티 (-2.0)
        animal_keywords = ["animal", "animals", "rat", "rats", "mouse", "mice", "in vitro", "cell culture"]
        for keyword in animal_keywords:
            if keyword in text or any(keyword in mesh for mesh in mesh_lower):
                score -= 2.0
                break
        
        return score

# -----------------------------
# BM25 Retriever
# -----------------------------
class BM25Retriever:
    """BM25-based keyword retrieval"""
    
    def __init__(self):
        try:
            from rank_bm25 import BM25Okapi
            self.BM25Okapi = BM25Okapi
            self.available = True
        except ImportError:
            print("!! rank_bm25 not available, using simple keyword matching")
            self.available = False
    
    def retrieve(self, queries: List[str], articles: List[Dict], top_k: int = 20) -> List[Tuple[int, float]]:
        """Retrieve articles using BM25"""
        if not articles:
            return []
        
        # Combine all queries
        combined_query = " ".join(queries)
        
        if self.available:
            # Use proper BM25
            corpus = [article["full_text"].lower().split() for article in articles]
            query_tokens = combined_query.lower().split()
            
            bm25 = self.BM25Okapi(corpus)
            scores = bm25.get_scores(query_tokens)
            
            # Get top results
            scored_articles = [(i, float(score)) for i, score in enumerate(scores)]
        else:
            # Fallback: simple keyword matching
            query_words = set(combined_query.lower().split())
            scored_articles = []
            
            for i, article in enumerate(articles):
                text_words = set(article["full_text"].lower().split())
                overlap = len(query_words.intersection(text_words))
                score = overlap / max(len(query_words), 1)
                scored_articles.append((i, score))
        
        # Sort by score and return top_k
        scored_articles.sort(key=lambda x: x[1], reverse=True)
        return scored_articles[:top_k]

# -----------------------------
# MedCPT Dense Retriever
# -----------------------------
class MedCPTRetriever:
    """Dense retrieval using MedCPT embeddings"""
    
    def __init__(self):
        try:
            from sentence_transformers import SentenceTransformer
            # MedCPT 모델 사용
            self.query_encoder = SentenceTransformer("ncbi/MedCPT-Query-Encoder")
            self.article_encoder = SentenceTransformer("ncbi/MedCPT-Article-Encoder")
            self.available = True
            print("@@ MedCPT encoders loaded successfully")
        except ImportError:
            print("!! sentence-transformers not available, using fallback")
            self.available = False
        except Exception as e:
            print(f"!! MedCPT not available ({e}), using general model")
            try:
                from sentence_transformers import SentenceTransformer
                # Fallback to biomedical model
                self.query_encoder = SentenceTransformer("pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb")
                self.article_encoder = self.query_encoder
                self.available = True
                print("@@ Fallback biomedical encoder loaded")
            except:
                    self.available = False
    
    def retrieve(self, queries: List[str], articles: List[Dict], top_k: int = 20) -> List[Tuple[int, float]]:
        """Retrieve articles using MedCPT dense embeddings"""
        if not self.available or not articles:
            return []
        
        try:
            # Encode query
            query_text = " ".join(queries)
            query_embedding = self.query_encoder.encode([query_text])
            
            # Encode articles
            article_texts = [article["full_text"] for article in articles]
            article_embeddings = self.article_encoder.encode(article_texts)
            
            # Calculate similarities
            from sklearn.metrics.pairwise import cosine_similarity
            similarities = cosine_similarity(query_embedding, article_embeddings)[0]
            
            # Get top results
            scored_articles = [(i, float(sim)) for i, sim in enumerate(similarities)]
            scored_articles.sort(key=lambda x: x[1], reverse=True)
            
            return scored_articles[:top_k]
            
        except Exception as e:
            print(f"!! MedCPT retrieval error: {e}")
            return []

# -----------------------------
# PubMedBERT Cross-Encoder
# -----------------------------
class PubMedBERTReranker:
    """Cross-encoder using PubMedBERT for reranking"""
    
    def __init__(self):
        try:
            from sentence_transformers import CrossEncoder
            # PubMedBERT 기반 크로스-인코더
            model_name = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract"
            self.model = CrossEncoder(model_name)
            self.available = True
            print(f"@@ PubMedBERT cross-encoder loaded: {model_name}")
        except ImportError:
            print("!! sentence-transformers not available")
            self.available = False
        except Exception as e:
            print(f"!! PubMedBERT not available ({e}), using fallback")
            try:
                from sentence_transformers import CrossEncoder
                # Fallback cross-encoder
                model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
                self.model = CrossEncoder(model_name)
                self.available = True
                print(f"@@ Fallback cross-encoder loaded: {model_name}")
            except:
                self.available = False
    
    def rerank(self, queries: List[str], articles: List[Dict], 
               candidate_indices: List[int], top_k: int = 5) -> List[Tuple[int, float]]:
        """Rerank candidates using PubMedBERT cross-encoder"""
        if not candidate_indices:
            return []
        
        if self.available:
            try:
                # Prepare query-article pairs
                query_text = " ".join(queries)
                pairs = []
                
                for idx in candidate_indices:
                    if idx < len(articles):
                        article_text = articles[idx]["full_text"][:512]  # Truncate for efficiency
                        pairs.append([query_text, article_text])
                
                if not pairs:
                    return []
                
                # Get scores
                scores = self.model.predict(pairs)
                scored_candidates = [(candidate_indices[i], float(score)) 
                                   for i, score in enumerate(scores)]
                
                # Sort and return top_k
                scored_candidates.sort(key=lambda x: x[1], reverse=True)
                return scored_candidates[:top_k]
                
            except Exception as e:
                print(f"!! Cross-encoder error: {e}")
        
        # Fallback: use simple text overlap
        query_words = set(" ".join(queries).lower().split())
        scored_candidates = []
        
        for idx in candidate_indices:
            if idx < len(articles):
                article_words = set(articles[idx]["full_text"].lower().split())
                overlap = len(query_words.intersection(article_words))
                score = overlap / max(len(query_words), 1)
                scored_candidates.append((idx, score))
        
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        return scored_candidates[:top_k]

# -----------------------------
# Evidence Level Scorer
# -----------------------------
class EvidenceLevelScorer:
    """CEBM 기반 근거 레벨 스코어링"""
    
    def __init__(self):
        # CEBM 근거 레벨 (높을수록 좋음)
        self.evidence_levels = {
            "meta-analysis": 5,
            "systematic review": 4,
            "randomized controlled trial": 3,
            "clinical trial": 2,
            "cohort study": 1,
            "case-control study": 1,
            "case series": 0,
            "case report": 0
        }
    
    def get_evidence_level(self, article: Dict) -> int:
        """Get evidence level score for article"""
        pub_types = [_safe_lower(pt) for pt in article.get("publication_types", [])]
        
        max_level = 0
        for pub_type in pub_types:
            for evidence_type, level in self.evidence_levels.items():
                if evidence_type in pub_type:
                    max_level = max(max_level, level)
        
        return max_level

# -----------------------------
# Margin Rule Selector
# -----------------------------
class MarginRuleSelector:
    """마진 규칙 기반 최종 문서 선택"""
    
    def __init__(self, margin_delta: float = 0.03):
        self.margin_delta = margin_delta
        self.evidence_scorer = EvidenceLevelScorer()
    
    def select_best_document(self, scored_candidates: List[Tuple[int, float]], 
                           articles: List[Dict]) -> Tuple[int, Dict]:
        """마진 규칙으로 최적 문서 선택"""
        if not scored_candidates:
            return -1, {}
        
        # Step 1: 최고 점수 찾기
        max_score = max(score for _, score in scored_candidates)
        
        # Step 2: 마진 내 동률 집합 구성
        tied_candidates = []
        for idx, score in scored_candidates:
            if max_score - score < self.margin_delta:
                tied_candidates.append((idx, score))
        
        print(f"@@ Margin rule: {len(tied_candidates)} candidates within margin δ={self.margin_delta}")
        
        # Step 3: 단일 후보면 바로 선택
        if len(tied_candidates) == 1:
            best_idx = tied_candidates[0][0]
            selection_info = {
                "selection_method": "margin_rule_single",
                "margin_delta": self.margin_delta,
                "max_score": max_score,
                "selected_score": tied_candidates[0][1]
            }
            return best_idx, selection_info
        
        # Step 4: 근거 레벨로 필터링
        best_evidence_level = -1
        evidence_filtered = []
        
        for idx, score in tied_candidates:
            article = articles[idx]
            evidence_level = self.evidence_scorer.get_evidence_level(article)
            
            if evidence_level > best_evidence_level:
                best_evidence_level = evidence_level
                evidence_filtered = [(idx, score, evidence_level)]
            elif evidence_level == best_evidence_level:
                evidence_filtered.append((idx, score, evidence_level))
        
        print(f"@@ Evidence filtering: {len(evidence_filtered)} candidates with level {best_evidence_level}")
        
        # Step 5: 여전히 동률이면 연도로 결정
        if len(evidence_filtered) == 1:
            best_idx = evidence_filtered[0][0]
            selection_info = {
                "selection_method": "margin_rule_evidence",
                "margin_delta": self.margin_delta,
                "max_score": max_score,
                "selected_score": evidence_filtered[0][1],
                "evidence_level": evidence_filtered[0][2]
            }
            return best_idx, selection_info
        
        # 최신 연도 선택
        latest_year = 0
        year_filtered = []
        
        for idx, score, evidence_level in evidence_filtered:
            article = articles[idx]
            try:
                year = int(article.get("year", "0"))
            except:
                year = 0
            
            if year > latest_year:
                latest_year = year
                year_filtered = [(idx, score, evidence_level, year)]
            elif year == latest_year:
                year_filtered.append((idx, score, evidence_level, year))
        
        # 최종 선택 (첫 번째를 선택)
        best_idx = year_filtered[0][0]
        selection_info = {
            "selection_method": "margin_rule_year",
            "margin_delta": self.margin_delta,
            "max_score": max_score,
            "selected_score": year_filtered[0][1],
            "evidence_level": year_filtered[0][2],
            "selected_year": year_filtered[0][3],
            "tied_count": len(year_filtered)
        }
        
        print(f"📅 Year filtering: selected {latest_year} (from {len(year_filtered)} candidates)")
        
        return best_idx, selection_info

# -----------------------------
# Main Enhanced RAG System
# -----------------------------
class EnhancedRAGSystem:
    """새로운 플로우의 Enhanced RAG system"""
    
    def __init__(self, margin_delta: float = 0.03):
        self.linearizer = TripleLinearizer()
        self.normalizer = MedicalNormalizer()
        self.pubmed_client = PubMedClient()
        self.bm25_retriever = BM25Retriever()
        self.medcpt_retriever = MedCPTRetriever()
        self.cross_encoder = PubMedBERTReranker()
        self.margin_selector = MarginRuleSelector(margin_delta)
        
        print("🚀 Enhanced RAG System (New Flow) initialized")
    
    def process_single_triple(self, disease: str, relation: str, drug: str, 
                            user_query: str = "", condition: str = "") -> Dict[str, Any]:
        """Process a single triple through the new RAG pipeline"""
        
        print(f"\n🔍 Processing: {disease} --[{relation}]--> {drug}")
        
        # Step 1: Triple Linearization
        print("(1) Step 1: Triple Linearization")
        search_queries = self.linearizer.create_search_queries(disease, relation, drug, user_query)
        linearized_sentence = self.linearizer.linearize_triple(disease, relation, drug)
        print(f"   Linearized: {linearized_sentence}")
        
        # Step 2: Term Normalization
        print("(2) Step 2: Medical Term Normalization")
        disease_terms = self.normalizer.get_mesh_terms(disease)
        drug_terms = self.normalizer.get_drug_synonyms(drug)
        print(f"   Disease terms: {disease_terms[:3]}...")
        print(f"   Drug terms: {drug_terms[:3]}...")
        
        # Step 3: PubMed Best Match 후보 수집
        print("(3) Step 3: PubMed Best Match Candidate Collection")
        
        # 첫 번째 시도: 임상 키워드 포함 쿼리
        pubmed_query = self.pubmed_client.create_pubmed_query(disease_terms, drug_terms)
        print(f"   Primary query: {pubmed_query[:100]}...")
        pmids = self.pubmed_client.search_pubmed_best_match(pubmed_query, max_results=200)
        
        # 결과가 부족하면 fallback 쿼리 사용
        if len(pmids) < 20:
            print(f"   Primary query returned only {len(pmids)} results, trying fallback...")
            fallback_query = self.pubmed_client.create_fallback_query(disease_terms, drug_terms)
            print(f"   Fallback query: {fallback_query[:100]}...")
            fallback_pmids = self.pubmed_client.search_pubmed_best_match(fallback_query, max_results=200)
            
            # 중복 제거하면서 결합
            seen_pmids = set(pmids)
            for pmid in fallback_pmids:
                if pmid not in seen_pmids:
                    pmids.append(pmid)
                    seen_pmids.add(pmid)
            
            used_query = f"Combined: {pubmed_query} + {fallback_query}"
        else:
            used_query = pubmed_query
        
        if not pmids:
            return {
                "success": False,
                "reason": "no_pubmed_results",
                "pubmed_query": used_query,
                "candidate_count": 0
            }
        
        articles = self.pubmed_client.fetch_articles_with_clinical_filter(pmids)
        print(f"   PubMed: {len(pmids)} PMIDs → {len(articles)} articles (Clinical filtered & sorted)")
        
        # 임상 관련성 상위 논문들 정보 출력
        if articles:
            top_scores = [(i, art["clinical_score"]) for i, art in enumerate(articles[:5])]
            print(f"   Top clinical scores: {[f'{i+1}:{score:.1f}' for i, score in top_scores]}")
        
        if not articles:
            return {
                "success": False,
                "reason": "no_articles_fetched",
                "pubmed_query": used_query,
                "pmid_count": len(pmids)
            }
        
        # Step 4: BM25 키워드 검색
        print("(4) Step 4: BM25 Keyword Retrieval")
        bm25_results = self.bm25_retriever.retrieve(search_queries, articles, top_k=20)
        print(f"   BM25: {len(bm25_results)} results")
        
        # Step 5: MedCPT Dense 검색
        print("(5) Step 5: MedCPT Dense Retrieval")
        dense_results = self.medcpt_retriever.retrieve(search_queries, articles, top_k=20)
        print(f"   MedCPT Dense: {len(dense_results)} results")
        
        # Step 6: BM25 + Dense 결합
        print("(6) Step 6: BM25 + Dense Combination")
        combined_candidates = set()
        
        # BM25 상위 20개 추가
        for idx, score in bm25_results:
            combined_candidates.add(idx)
        
        # Dense 상위 20개 추가
        for idx, score in dense_results:
            combined_candidates.add(idx)
        
        candidate_indices = list(combined_candidates)
        print(f"   Combined: {len(candidate_indices)} unique candidates from BM25({len(bm25_results)}) + Dense({len(dense_results)})")
        
        # Step 7: PubMedBERT Cross-Encoder 리랭킹
        print("(7) Step 7: PubMedBERT Cross-Encoder Reranking")
        reranked_results = self.cross_encoder.rerank(
            search_queries, articles, candidate_indices, top_k=5
        )
        print(f"   Cross-encoder: {len(reranked_results)} reranked articles")
        
        if not reranked_results:
            return {
                "success": False,
                "reason": "no_reranked_results",
                "pubmed_query": used_query,
                "candidate_count": len(articles)
            }
        
        # Step 8: 마진 규칙 기반 최종 선택
        print("(8) Step 8: Margin Rule Final Selection")
        best_idx, selection_info = self.margin_selector.select_best_document(reranked_results, articles)
        
        if best_idx == -1:
            return {
                "success": False,
                "reason": "margin_rule_failed",
                "pubmed_query": used_query,
                "reranked_count": len(reranked_results)
            }
        
        best_article = articles[best_idx]
        print(f"@@ Selected: {best_article['title'][:60]}...")
        print(f"   Method: {selection_info['selection_method']}")
        print(f"   Journal: {best_article['journal']} ({best_article['year']})")
            
        return {
            "success": True,
            "best_paper": best_article,
            "selection_info": {
                **selection_info,
                "linearized_sentence": linearized_sentence,
                "search_queries": search_queries,
                "pubmed_query": used_query
            },
            "pipeline_stats": {
                "initial_pmids": len(pmids),
                "fetched_articles": len(articles),
                "bm25_candidates": len(bm25_results),
                "dense_candidates": len(dense_results),
                "combined_candidates": len(candidate_indices),
                "reranked_candidates": len(reranked_results)
            }
        }

# -----------------------------
# Pipeline Interface Function (실시간 RAG 처리)
# -----------------------------
def get_treatment_context_from_triple(triple_result: Dict, openai_api_key: str = None, max_papers: int = 5) -> str:
    """
    파이프라인에서 호출할 수 있는 함수 (실시간 RAG 처리)
    Triple 결과를 받아서 실시간으로 PubMed 검색 및 RAG 처리를 수행
    """
    if triple_result.get('status') != 'success':
        return "No treatment information available due to triple extraction failure."
    
    try:
        selected_triple = triple_result['selected_triple']
        condition = triple_result['condition']
        user_query = triple_result.get('user_query', '')
        
        disease = selected_triple['start_node']['name']
        relation = selected_triple['relation']
        drug = selected_triple['end_node']['name']
        
        print(f"🔍 실시간 RAG 처리: {condition} - {disease} --[{relation}]--> {drug}")
        
        # RAG 시스템 초기화
        rag_system = EnhancedRAGSystem(margin_delta=0.03)
        
        # 실시간 RAG 처리 실행
        rag_result = rag_system.process_single_triple(
            disease=disease,
            relation=relation,
            drug=drug,
            user_query=user_query,
            condition=condition
        )
        
        if rag_result["success"]:
            paper = rag_result["best_paper"]
            
            # 논문 정보를 문맥으로 구성
            context = (
                f"**Medication: {drug}**\n"
                f"Title: {paper['title']}\n"
                f"Abstract: {paper['abstract']}"
            )
            
            print(f"@@ RAG 성공: {paper['title'][:60]}...")
            return context
        else:
            print(f"X RAG 실패: {rag_result['reason']}")
            return f"No relevant papers found for {drug} treatment of {condition}."
            
    except Exception as e:
        print(f"!! Error in real-time RAG processing: {e}")
        return "Professional medication evaluation and treatment should be considered as part of comprehensive care."


# -----------------------------
# Main Pipeline
# -----------------------------
def main():
    """Main pipeline: Load triples and process each with new RAG flow"""
    import subprocess
    
    print("@@ Starting Enhanced RAG Pipeline (New Flow)")
    print("="*80)
    
    # Step 1: Generate fresh recommendations
    print("📋 Step 1: Generating fresh recommendations...")
    try:
        result = subprocess.run([
            sys.executable, "src/kg2subgraph/label2triple.py"
        ], capture_output=True, text=True, cwd="/home/dksk1090/temp_kg")
        
        if result.returncode != 0:
            print(f"X label2triple.py failed: {result.stderr}")
            return
        print("@@ Fresh recommendations generated")
    except Exception as e:
        print(f"X Error running label2triple.py: {e}")
        return
    
    # Step 2: Load recommendations
    print("\n📂 Step 2: Loading recommendations...")
    input_file = "output_dir/test_final_recommendations.json"
    
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            recommendations = json.load(f)
        print(f"@@ Loaded {len(recommendations)} condition recommendations")
    except Exception as e:
        print(f"X Error loading recommendations: {e}")
        return
    
    # Step 3: Initialize RAG system
    print("\n(3) Step 3: Initializing New Flow RAG System...")
    rag_system = EnhancedRAGSystem(margin_delta=0.03)
    
    # Step 4: Process each condition
    print("\n(4) Step 4: Processing each condition...")
    results = []
    
    for item in recommendations:
        if item.get('status') != 'success' or 'selected_triple' not in item:
            print(f"!! Skipping {item.get('condition', 'unknown')}: no valid triple")
            continue
        
        condition = item['condition']
        user_query = item['user_query']
        selected_triple = item['selected_triple']
        
        print(f"\n{'='*60}")
        print(f"@@ Processing condition: {condition}")
        print(f"@@ User query: {user_query}")
        print(f"@@ Selected triple: {selected_triple['start_node']['name']} --[{selected_triple['relation']}]--> {selected_triple['end_node']['name']}")
        
        # Process with new RAG flow
        try:
            rag_result = rag_system.process_single_triple(
                disease=selected_triple['start_node']['name'],
                relation=selected_triple['relation'],
                drug=selected_triple['end_node']['name'],
                user_query=user_query,
                condition=condition
            )
            
            # Combine with original data
            final_result = {
                "condition": condition,
                "user_query": user_query,
                "original_triple": selected_triple,
                "rag_result": rag_result,
                "timestamp": datetime.now().isoformat()
            }
            
            results.append(final_result)
            
            if rag_result["success"]:
                paper = rag_result["best_paper"]
                print(f"@@ Success! Found: {paper['title'][:60]}...")
                print(f"   Method: {rag_result['selection_info']['selection_method']}")
            else:
                print(f"X Failed: {rag_result['reason']}")
            
        except Exception as e:
            print(f"X Error processing {condition}: {e}")
            error_result = {
                "condition": condition,
                "user_query": user_query,
                "original_triple": selected_triple,
                "rag_result": {"success": False, "reason": f"Error: {str(e)}"},
                "timestamp": datetime.now().isoformat()
            }
            results.append(error_result)
    
    # Step 5: Save results
    print(f"\n@@ Step 5: Saving results...")
    output_path = "output_dir/RAG/enhanced_rag_results.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"@@ Results saved to: {output_path}")
    
    # Step 6: Summary
    print(f"\n@@ Final Summary:")
    print(f"   Total conditions processed: {len(results)}")
    
    successful = [r for r in results if r["rag_result"]["success"]]
    print(f"   Successful retrievals: {len(successful)}")
    
    # 선택 방법별 통계
    selection_methods = {}
    for result in successful:
        method = result["rag_result"]["selection_info"]["selection_method"]
        selection_methods[method] = selection_methods.get(method, 0) + 1
    
    print(f"\n@@ Selection methods used:")
    for method, count in selection_methods.items():
        print(f"   {method}: {count}")
    
    print(f"\n@@ Results by condition:")
    for result in results:
        condition = result["condition"]
        drug = result["original_triple"]["end_node"]["name"]
        
        if result["rag_result"]["success"]:
            paper = result["rag_result"]["best_paper"]
            method = result["rag_result"]["selection_info"]["selection_method"]
            print(f"   @@ {condition}: {drug}")
            print(f"      📄 {paper['title'][:50]}...")
            print(f"      🎯 Method: {method} | 📖 {paper['journal']} ({paper['year']})")
        else:
            reason = result["rag_result"]["reason"]
            print(f"   X {condition}: {drug} - {reason}")
        print()
    
    print("🎉 Enhanced RAG Pipeline (New Flow) completed!")

if __name__ == "__main__":
    main()
