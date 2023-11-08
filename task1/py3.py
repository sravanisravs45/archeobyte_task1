# check 
test_eg=['what is the current marketprice of petroleum?', 
		'who is Oswald?', 'why are you here idiot ?']
test_data =from_tensor_slices((test_eg, [0]*len(test_eg)))
# wrap test data into BERT format
test_data = (test_data.map(to_feature_map_bert).batch(1))
preds = model.predict(test_data)
print(preds)
['Insincere' if pred >=0.5 else 'Sincere' for pred in preds]