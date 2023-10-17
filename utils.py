import numpy as np
from fuzzywuzzy import process


def cache_query_mask(ner_model,payload,outputs=None):
  masked = []
  maps  = []

  for j in range(len(payload)):
    if outputs==None:
      response = ner_model(payload[i])
    else:
      response = outputs[j]
    running, qns, map = 0, payload[j], []
    print(response)
    for i in range(len(response)):
      if response[i]['entity_group'] == 'ORG':
        qns = qns[:running+response[i]['start']]+response[i]['entity_group']+qns[running+response[i]['end']:]
        running += len(response[i]['entity_group'])-len(response[i]['word'])
        map.append({'ORG_'+str(i+1):response[i]['word']})
    masked.append(qns)
    maps.append(map)
  return masked, maps


def cache_sql_mask(strings, companies):
    modified_strings = []

    for string in strings:
        org_counter = 1
        company_positions = []

        for company in companies:
            position = string.find(company)
            if position != -1:
                company_positions.append((position, company))

        company_positions.sort()

        for _, company in company_positions:
            string = string.replace(company, f'ORG_{org_counter}')
            org_counter += 1

        modified_strings.append(string)

    return modified_strings


def cosine_similarity(queries, embeddings):
    queries_norm = queries / np.linalg.norm(queries, axis=1, keepdims=True)
    embeddings_norm = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    dot_product = np.dot(queries_norm, embeddings_norm.T)

    return dot_product

def cache_embed(embed_model,cached_qns,instruction = "Represent the financial question for clustering:"):
  embeded_cache = []
  for i in cached_qns:
    embeded_cache.append(embed_model.encode([[instruction,i]])[0])
  return embeded_cache


def find_top_k_similar(user_queries, embeddings, k):
    normalized_embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized_queries = user_queries / np.linalg.norm(user_queries, axis=1, keepdims=True)
    similarities = cosine_similarity(normalized_queries, normalized_embeddings)
    top_k_indices = np.argpartition(-similarities, k, axis=1)[:, :k]
    rows = np.arange(len(user_queries))[:, None]
    sorted_order = np.argsort(-similarities[rows, top_k_indices], axis=1)
    top_k_indices = np.take_along_axis(top_k_indices, sorted_order, axis=1)
    top_k_similarities = np.take_along_axis(similarities, top_k_indices, axis=1)
    result = [[(index, similarity) for index, similarity in zip(indices, sims)]
              for indices, sims in zip(top_k_indices, top_k_similarities)]

    return result

def map_usr_qry_sql(top_k,masked_user_maps,masked_answers):
  if top_k[0][0][1]<0.91:
    print('Below Threshold')
    return None
  else:
    result = masked_answers[top_k[0][0][0]]
    for i in masked_user_maps[0]:
      org = list(i.keys())[0]

      result = result.replace(org,i[org])
    return result, masked_user_maps[0]


def get_closest_match(entity, candidates):
    closest_match, score = process.extractOne(entity, candidates)
    return closest_match, score

def map_entity_to_table(entities, db_entity_names, db_entity_tickers, inter_dict):
    normalized_entities = []
    normalized_tickers = []

    for entity in entities:
        max_name_score = -1
        max_ticker_score = -1
        chosen_name = ""
        chosen_ticker = ""

        for db_name in db_entity_names:
            synonyms = inter_dict.get(db_name, [db_name])
            closest_synonym, synonym_score = get_closest_match(entity, synonyms)

            if synonym_score > max_name_score:
                max_name_score = synonym_score
                chosen_name = db_name

        closest_ticker, ticker_score = get_closest_match(entity, db_entity_tickers)

        if ticker_score > max_ticker_score:
            max_ticker_score = ticker_score
            chosen_ticker = closest_ticker

        if max_name_score >= max_ticker_score:
            normalized_entities.append(chosen_name)
            index_of_chosen_name = db_entity_names.index(chosen_name)
            normalized_tickers.append(db_entity_tickers[index_of_chosen_name])
        else:
            index_of_chosen_ticker = db_entity_tickers.index(chosen_ticker)
            normalized_entities.append(db_entity_names[index_of_chosen_ticker])
            normalized_tickers.append(chosen_ticker)

    return normalized_entities, normalized_tickers

