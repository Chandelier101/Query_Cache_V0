from InstructorEmbedding import INSTRUCTOR
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline

def ner():
    tokenizer = AutoTokenizer.from_pretrained('flair/ner-english-ontonotes-large')
    model = AutoModelForTokenClassification.from_pretrained('flair/ner-english-ontonotes-large')
    ner = pipeline("ner", model=model, tokenizer=tokenizer)
    return ner

# def cache_query_mask(API_URL,headers,payload,outputs=None):
#   masked = []
#   maps  = []
#   API_URL = "https://api-inference.huggingface.co/models/flair/ner-english-ontonotes-large"
#   headers = {"Authorization": "Bearer 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"}
#   for j in range(len(payload)):
#     if outputs==None:
#       response = requests.post(API_URL, headers=headers, json=payload[j])
#       response = response.json()
#     else:
#       response = outputs[j]
#     running, qns, map = 0, payload[j], []
#     print(response)
#     for i in range(len(response)):

#       if response[i]['entity_group'] == 'ORG':
#         qns = qns[:running+response[i]['start']]+response[i]['entity_group']+qns[running+response[i]['end']:]
#         running += len(response[i]['entity_group'])-len(response[i]['word'])
#         map.append({'ORG_'+str(i+1):response[i]['word']})
#     masked.append(qns)
#     maps.append(map)
#   return masked, maps



# embed_model = INSTRUCTOR('hkunlp/instructor-large')
