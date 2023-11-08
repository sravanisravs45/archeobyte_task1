# split into train and validation
train_df, remaining = train_test_split(df, train_size=0.01,
									stratify=df.target.values)
valid_df, _ = train_test_split(remaining, train_size=0.001,
							stratify=remaining.target.values)
train_df.shape, valid_df.shape

# import for processing dataset
from tf.data.Dataset import from_tensor_slices
from tf.data.experimental import AUTOTUNE

# convert dataset into tensor slices
with tf.device('/cpu:0'):
train_data =from_tensor_slices((train_df.question_text.values,
												train_df.target.values))
valid_data = from_tensor_slices((valid_df.question_text.values,
												valid_df.target.values))
	
for text, label in train_data.take(2):
	print(text)
	print(label)
	
label_list = [0, 1] # Label categories
max_seq_length = 128 # maximum length of input sequences
train_batch_size = 32

# Get BERT layer and tokenizer:
bert_layer = hub.KerasLayer(
"https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/2",
							trainable=True)
vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
tokenizer = tokenization.FullTokenizer(vocab_file, do_lower_case)

# example
# convert to tokens ids and 
tokenizer.convert_tokens_to_ids(
tokenizer.wordpiece_tokenizer.tokenize('how are you?'))

# convert the dataset into the format required by BERT i.e we convert the row into
# input features (Token id, input mask, input type id ) and labels

def convert_to_bert_feature(text, label, label_list=label_list, 
			max_seq_length=max_seq_length, tokenizer=tokenizer):
example = classifier_data_lib.InputExample(guid = None,
											text_a = text.numpy(), 
											text_b = None, 
											label = label.numpy())
feature = classifier_data_lib.convert_single_example(0, example, label_list,
									max_seq_length, tokenizer)

return (feature.input_ids, feature.input_mask, feature.segment_ids, 
		feature.label_id)

# wrap the dataset around the python function in order to use the tf
# datasets map function
def to_bert_feature_map(text, label):

input_ids, input_mask, segment_ids, label_id = tf.py_function(
	convert_to_bert_feature,
	inp=[text, label],
	Tout=[tf.int32, tf.int32, tf.int32, tf.int32])

# py_func doesn't set the shape of the returned tensors.
input_ids.set_shape([max_seq_length])
input_mask.set_shape([max_seq_length])
segment_ids.set_shape([max_seq_length])
label_id.set_shape([])

x = {
		'input_word_ids': input_ids,
		'input_mask': input_mask,
		'input_type_ids': segment_ids
	}
return (x, label_id)
with tf.device('/cpu:0'):
# train
train_data = (train_data.map(to_bert_feature_map,
							num_parallel_calls=AUTOTUNE)
						#.cache()
						.shuffle(1000)
						.batch(32, drop_remainder=True)
						.prefetch(AUTOTUNE))

# valid
valid_data = (valid_data.map(to_bert_feature_map,
							num_parallel_calls=AUTOTUNE)
						.batch(32, drop_remainder=True)
						.prefetch(AUTOTUNE)) 

# example format train and valid data
print("train data format",train_data.element_spec)
print("validation data format",valid_data.element_spec)