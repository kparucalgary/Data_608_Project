# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 09:22:07 2026

@author: dicke
"""


from typing import TypedDict, Literal, List
from langgraph.graph import StateGraph, END
import numpy as np
from search_test import semantic_search
from on_demand_pipeline import run_ondemand_pipeline
from langchain_ollama import ChatOllama

# Initialize Gemma 3
#llm = ChatOllama(model="gemma3:4b", temperature=0)
llm = ChatOllama(model="qwen2.5:7b-instruct-q4_K_M", temperature=0)
#from symbolic_module import build_symbolic_query
#from retrieval_module import retrieve_papers, retrieve_papers_expanded

class GraphState(TypedDict):
    mode: str
    numpapers: int
    threshold: float
    query: str
    queryret0: str
    queryret1: str
    queryret2: str
    queryret3: str
    
    symbolic_query: str
    titles: List[str]
    scores: List[float]
    abstracts: List[str]
    links: List[str]
    
    retry_count: int
    message: str
    final_output: str
    modified_query: str
    


def startNode(state: GraphState) -> GraphState:
    return {
        **state,
        "retry_count": 0,
        "retrieved_papers": [],
        "scores": [],
        "modified_query": "",
        "message": "",
        "queryret0": "",
        "queryret1": "",
        "queryret2": "",
        "queryret3": "",
    }


def queryNode(state: GraphState) -> GraphState:
    if state["mode"] == "mvp":
        # skip symbolic transformation
        modified_query = state["query"]
    if state["mode"] == "llm":
        modified_query = llm.invoke(''''You are a linguistics specialist with 
                                    20+ years of experience tasked with giving 
                                    an interpretation of a query to aid in a 
                                    semantic search over academic papers to 
                                    find the best and most related results. 
                                    You are to return only the modified query 
                                    to retreive the best possible results, 
                                    with no newlines or explanations of your 
                                    modification. Here is the original query: 
                                    ''' + state["query"])
        state["threshold"] = state["threshold"]
        state["numpapers"] = state["numpapers"]
        modified_query = modified_query.content
    state['queryret0'] = modified_query
    
    return {**state, "modified_query": modified_query}



def retrieveNode(state: GraphState) -> GraphState:
    retry = state["retry_count"]
    query = state["modified_query"]
        
    if retry == 0:
        results = semantic_search(query, top_k=state["numpapers"])
        state['links'] = [i['link'] for i in results]
    else:
        # progressively broader retrieval
        if retry == 1:
            if state['mode'] == 'llm':
                queryret = llm.invoke(''''You are a linguistics specialist with 
                                        20+ years of experience tasked with modifying 
                                        a query to aid in a 
                                        semantic search,
                                        over an academic paper database to 
                                        find the best and most related results. 
                                        You are to return only the modified query 
                                        to retreive the best possible results, 
                                        with no newlines or explanations of your 
                                        modification. Do not rely on search 
                                        keywords such as AND or OR. The goal 
                                        is not to find papers on semantic 
                                        search unless that is the query given 
                                        at the end of this prompt. The goal 
                                        is to find new papers not found by the 
                                        initial search. Here is the original 
                                        prompt: ''' + query + ''' and first 
                                        modified query: 
                                        ''' + state['queryret0'])
                queryret = queryret.content
            else:
                queryret = query + ' application'
            state['queryret1'] = queryret
        elif retry == 2:
            if state['mode'] == 'llm':
                queryret = llm.invoke(''''You are a linguistics specialist with 
                                        20+ years of experience tasked with modifying 
                                        a query to aid in a 
                                        semantic search,
                                        over an academic paper database to 
                                        find the best and most related results. 
                                        You are to return only the modified query 
                                        to retreive the best possible results, 
                                        with no newlines or explanations of your 
                                        modification. Do not rely on search 
                                        keywords such as AND or OR. The goal 
                                        is not to find papers on semantic 
                                        search unless that is the query given 
                                        at the end of this prompt. The goal 
                                        is to find new papers not found by the 
                                        initial search.Here is the original 
                                        prompt: ''' + query + ''' and first 
                                        modified query: 
                                        ''' + state['queryret0'] + ''' and here is 
                                        the second modified query: ''' +
                                        state['queryret1'])
                queryret = queryret.content
            else:
                queryret = query + ' results'
            state['queryret2'] = queryret
        elif retry == 3:
            
            if state['mode'] == 'llm':
                queryret = llm.invoke(''''You are a linguistics specialist with 
                                        20+ years of experience tasked with modifying 
                                        a query to aid in a 
                                        semantic search,
                                        over an academic paper database to 
                                        find the best and most related results. 
                                        You are to return only the modified query 
                                        to retreive the best possible results, 
                                        with no newlines or explanations of your 
                                        modification. Do not rely on search 
                                        keywords such as AND or OR. The goal 
                                        is not to find papers on semantic 
                                        search unless that is the query given 
                                        at the end of this prompt. The goal 
                                        is to find new papers not found by the 
                                        initial search. Here is the original 
                                        prompt: ''' + query + ''' and first 
                                        modified query: 
                                        ''' + state['queryret0'] + ''' and here is 
                                        the second modified query: ''' + 
                                        state['queryret1'] + ''' and finally here 
                                        is the third modified query: ''' + 
                                        state['queryret2'])
                queryret = queryret.content
            else:
                queryret = query + ' success'
            state['queryret3'] = queryret
            
        
        results = run_ondemand_pipeline(queryret)
        results = results['results']
        state['links'] = [i['url'] for i in results]

    state['titles'] = [i['title'] for i in results]
    state['abstracts'] = [i['abstract'] for i in results]
    state['scores'] = [i['score'] for i in results]
    
    return {
        **state,
    }

def checkRelevantPapers(state: GraphState) -> GraphState:
    threshold = state["threshold"]
    numpapers = state["numpapers"]
    scores = np.array(state["scores"])
    num_above = np.sum(scores >= threshold) # number of papers above threshold
    
    if num_above >= numpapers:
        return {**state, "message": "Enough Papers"}
    
    retry = state["retry_count"]
    
    if retry >= 3:
        return {**state, "message": "Max Retries Reached"}
    
    return {**state, "message": f"Insufficient Papers {retry + 1}"}
    
def routeNode(state: GraphState) -> str:
    if state["message"] == "Enough Papers":
        return "synthesize"
    
    if state["message"] == "Max Retries Reached":
        return "synthesize"
    
    return "retry"
    
def retryNode(state: GraphState) -> GraphState:
    return {
        **state,
        "retry_count": state["retry_count"] + 1
    }



def synthesisNode(state: GraphState) -> GraphState:
    titles = state["titles"]
    scores = state["scores"]
    abstracts = state["abstracts"]
    links = state["links"]
    threshold = state["threshold"]
    k = state["numpapers"]
    
    # Pair papers with scores
    paper_score_pairs = list(zip(titles, scores, links, abstracts))

    # Filter by threshold
    filtered = [
        (title, score, link, abstract)
        for title, score, link, abstract in paper_score_pairs
        if score >= threshold
    ]

    # Sort by score (descending)
    sorted_papers = sorted(
        filtered,
        key=lambda x: x[1],
        reverse=True
    )

    # Select top-k
    if len(filtered) == 0:
        return {**state, "final_output": [{'title': 'No results found', 'link': '', 'abstract': ''}]}
    if k > len(filtered):
        top_papers = sorted_papers[:k]
    else:
        top_papers = sorted_papers

    # Separate again
    #selected_papers = [p for p, _, _, _ in top_papers]

    # --- Mode handling ---
    output = []
    for i in top_papers:
        output.append({'title': i[0], 'link': i[2], 'abstract': i[3]})
    return {
        **state,
        "final_output": output
    }


builder = StateGraph(GraphState)

builder.add_node("start", startNode)
builder.add_node("query", queryNode)
builder.add_node("retrieve", retrieveNode)
builder.add_node("check", checkRelevantPapers)
builder.add_node("retry", retryNode)
builder.add_node("synthesize", synthesisNode)

builder.set_entry_point("start")

builder.add_edge("start", "query")
builder.add_edge("query", "retrieve")
builder.add_edge("retrieve", "check")

builder.add_conditional_edges(
    "check",
    routeNode,
    {
        "retry": "retry",
        "synthesize": "synthesize"
    }
)

builder.add_edge("retry", "retrieve")
builder.set_finish_point("synthesize")

graph = builder.compile()

def run_pipeline(query: str, mode: str, numpapers: int, threshold: float):
    initial_state = {
        "mode": mode,
        "query": query,
        "numpapers": numpapers,
        "threshold": threshold,
    }

    result = graph.invoke(initial_state)

    return result["final_output"]