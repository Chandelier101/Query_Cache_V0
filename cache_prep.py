import pickle
import utils
import model

questions = [
"What is the market cap of Apple Inc.?",
"Which company has the highest net profit margin?",
"Compare the revenue per Share between Alphabet Inc. and Meta Platforms, Inc.",
"What is Tesla Inc.'s ev/ebibta ratio?",
"What is the Forward P/E ratio of Amazon.com, Inc.?",
"What is the difference in revenue per Employee between Alphabet Inc. and Meta Platforms, Inc.?",
"Which company has the highest Dividend Yield?",
"What is the Operating margin of NVIDIA Corporation?",
"Which company has the highest revenue 3Y CAGR?",
"Compare the Gross Profit per Employee between Meta Platforms, Inc. and Nasdaq, Inc.",
"Which companies have a P/E ratio less than 20?",
"What is the Return on Assets (ROA) for Netflix, Inc.?",
"Which company has the lowest Payout Ratio?",
"What is the Net Income per Employee for Meta Platforms, Inc.?",
"Which company has the highest Price to Sales (P/S) ratio?",
"What is the EBITDA margin for Alphabet Inc.?",
"Compare the Price to Earnings to Growth (PEG) ratio between Google and Meta.",
"Which company has the lowest revenue 5Y CAGR?",
"What is the Pre-Tax Profit margin of Tesla, Inc.?",
"What is the Total Enterprise Value (TEV) of Nasdaq, Inc.?",
"Which company has the highest EV/Sales ratio?",
"What is the Return on Equity (ROE) for Amazon.com, Inc.?"
]

answers = [
'''SELECT "Market Cap" FROM Aggregate_Fundamentals_Data_Basic WHERE Company = 'Apple Inc.';''',
'''SELECT Company FROM Aggregate_Fundamentals_Data_Profitability ORDER BY "Net Profit margin" DESC LIMIT 1;''',
'''SELECT Company, "revenue per Share" FROM Aggregate_Fundamentals_Data_Per Share WHERE Company IN ('Alphabet Inc.', 'Meta Platforms, Inc.');''',
'''SELECT "EV/EBITDA" FROM Aggregate_Fundamentals_Data_Valuation WHERE Company = 'Tesla, Inc.';''',
'''SELECT "Forward P/E" FROM Aggregate_Fundamentals_Data_Forward Valuation WHERE Company = 'Amazon.com, Inc.';''',
'''SELECT Company, "revenue per Employee" FROM Aggregate_Fundamentals_Data_Employees WHERE Company IN ('Alphabet Inc.', 'Meta Platforms, Inc.');''',
'''SELECT Company FROM Aggregate_Fundamentals_Data_Dividends ORDER BY "Dividend Yield" DESC LIMIT 1;''',
'''SELECT "Operating margin" FROM Aggregate_Fundamentals_Data_Profitability WHERE Company = 'NVIDIA Corporation';''',
'''SELECT Company FROM Aggregate_Fundamentals_Data_Growth ORDER BY "revenue 3Y CAGR" DESC LIMIT 1;''',
'''SELECT Company, "Gross Profit per Employee" FROM Aggregate_Fundamentals_Data_Employees WHERE Company IN ('Meta Platforms, Inc.', 'Nasdaq, Inc.');''',
'''SELECT Company FROM Aggregate_Fundamentals_Data_Valuation WHERE "P/E" < 20;''',
'''SELECT "Return on Assets (ROA)" FROM Aggregate_Fundamentals_Data_Capital Efficiency WHERE Company = 'Netflix, Inc.';''',
'''SELECT Company FROM Aggregate_Fundamentals_Data_Dividends ORDER BY "Payout Ratio" ASC LIMIT 1;''',
'''SELECT "Net Income per Employee" FROM Aggregate_Fundamentals_Data_Employees WHERE Company = 'Meta Platforms, Inc.';''',
'''SELECT Company FROM Aggregate_Fundamentals_Data_Valuation ORDER BY "P/S" DESC LIMIT 1;''',
'''SELECT "EBITDA margin" FROM Aggregate_Fundamentals_Data_Profitability WHERE Company = 'Alphabet Inc.';''',
'''SELECT Company, "P/E" FROM Aggregate_Fundamentals_Data_Valuation WHERE Company IN ('Alphabet Inc.', 'Meta Platforms, Inc.');''',
'''SELECT Company FROM Aggregate_Fundamentals_Data_Growth ORDER BY "revenue 5Y CAGR" ASC LIMIT 1;''',
'''SELECT "Pre-Tax Profit margin" FROM Aggregate_Fundamentals_Data_Profitability WHERE Company = 'Tesla, Inc.';''',
'''SELECT "Total Enterprise Value (TEV)" FROM Aggregate_Fundamentals_Data_Basic WHERE Company = 'Nasdaq, Inc.';''',
'''SELECT Company FROM Aggregate_Fundamentals_Data_Valuation ORDER BY "EV/Sales" DESC LIMIT 1;''',
'''SELECT "Return on Equity (ROE)" FROM Aggregate_Fundamentals_Data_Capital Efficiency WHERE Company = 'Amazon.com, Inc.';'''
]

alias_dict = {'Apple Inc.': ['Apple Inc.'],
 'Alphabet Inc.': ['Alphabet Inc.', 'Google Inc.'],
 'Amazon.com, Inc.': ['Amazon.com, Inc.'],
 'NVIDIA Corporation': ['NVIDIA Corporation'],
 'Meta Platforms, Inc.': ['Meta Platforms, Inc.',
  'Meta Inc.',
  'Facebook Inc.'],
 'Netflix, Inc.': ['Netflix, Inc.'],
 'Nasdaq, Inc.': ['Nasdaq, Inc.'],
 'Tesla, Inc.': ['Tesla, Inc.']}


# masked_answers = cache_sql_mask(answers, list(alias_dict.keys()))
# db_entity_names = list(alias_dict.keys())
# db_entity_tickers = ['AAPL', 'GOOGL', 'AMZN', 'NVDA', 'META', 'NFLX', 'NDAQ', 'TSLA']
# masked_qns, masked_maps = cache_query_mask('','',questions)
# embeded_cache = cache_embed(embed_model,masked_qns)
# user_query = "Give me the ebitda margins for Onion .Inc"

def preprocess_and_save():
    embed_model = model.INSTRUCTOR('hkunlp/instructor-large')
    masked_answers = utils.cache_sql_mask(answers, list(alias_dict.keys()))
    db_entity_names = list(alias_dict.keys())
    db_entity_tickers = ['AAPL', 'GOOGL', 'AMZN', 'NVDA', 'META', 'NFLX', 'NDAQ', 'TSLA']
    ner_model = model.ner()
    masked_qns, masked_maps = utils.cache_query_mask(ner_model,questions)
    embeded_cache = utils.cache_embed(embed_model,masked_qns)

    preprocessed_data = {'masked_answers':masked_answers,
                         'db_entity_tickers': db_entity_tickers,
                         'db_entity_names':db_entity_names,
                         'masked_qns':masked_qns,
                         'masked_maps':masked_maps,
                         'embeded_cache':embeded_cache}
    with open('preprocessed_cache.pkl', 'wb') as f:
        pickle.dump(preprocessed_data, f)
        
    del ner_model
    del embed_model
if __name__ == "__main__":
    preprocess_and_save()