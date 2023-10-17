from flask import Flask, request, jsonify
import pickle
import utils
import model
from cache_prep import preprocess_and_save

app = Flask(__name__)


preprocess_and_save()
# Load the preprocessed cache data
with open('preprocessed_cache.pkl', 'rb') as f:
    preprocessed_cache = pickle.load(f)

# Load embedding model
embed_model = model.INSTRUCTOR('hkunlp/instructor-large')
ner_model  = model.ner()

# Full pipeline
def full_pipe(user_query,k=1):
  masked_user_qns, masked_user_maps = utils.cache_query_mask(ner_model,[user_query])
  user_query_embed = utils.cache_embed(embed_model,masked_user_qns)
  top_k = utils.find_top_k_similar(user_query_embed,preprocessed_cache.embeded_cache,k)
  print("Masked input query",masked_user_qns)
  print("Masked input maps",masked_user_maps)
  print(preprocessed_cache.masked_answers[top_k[0][0][0]])
  for i in range(len(masked_user_maps[0])):
    map = list(masked_user_maps[0][i].keys())[0]
    masked_user_maps[0][i][map] = utils.map_entity_to_table([masked_user_maps[0][i][map]], 
                                                            preprocessed_cache.db_entity_names, 
                                                            preprocessed_cache.db_entity_tickers,
                                                            preprocessed_cache.alias_dict)[0][0]
  print(masked_user_maps)
  print("Instruct Embedding Similarity:",top_k)
  return utils.map_usr_qry_sql(top_k,masked_user_maps,preprocessed_cache.masked_answers)


@app.route('/cache-swap', methods=['POST'])
def cache_swap():
    try:
        data = request.get_json()
        user_query = data.get('user_query')
        
        if not user_query:
            return jsonify({"error": "User question is required"}), 400
        
        response = full_pipe(user_query)
        
        if response:
            return jsonify({"mapped_sql": response})
        else:
            return jsonify({"error": "Couldn't map the query"}), 400
            
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5000)
