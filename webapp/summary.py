from transformers import pipeline, TFAutoModelForSeq2SeqLM, AutoTokenizer
import tensorflow as tf
tf.get_logger().setLevel('INFO')


def bert_model(ARTICLE):
	summ = pipeline("summarization")
	a = summ(ARTICLE, max_length=300, min_length=30, do_sample=False)
	return a[0].get('summary_text')

def t5_model(ARTICLE):
	model = TFAutoModelForSeq2SeqLM.from_pretrained("t5-base")
	tokenizer = AutoTokenizer.from_pretrained("t5-base")

	# T5 uses a max_length of 512 so we cut the article to 512 tokens.
	inputs = tokenizer.encode("summarize: " + ARTICLE, return_tensors="tf", max_length=512)
	outputs = model.generate(inputs,
							max_length=300,
							min_length=30,
							length_penalty=2.0,
							num_beams=4,
							early_stopping=True)
	res = tokenizer.decode(outputs[0])

	return res