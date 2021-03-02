# FinBERT: Financial Sentiment Analysis with BERT

## Analysis of FinBERT

We tested the sentiment prediction accuracy of FinBERT with three datasets:  
1. Financial PhraseBank from [Malo et al. (2014)](https://www.researchgate.net/publication/251231107_Good_Debt_or_Bad_Debt_Detecting_Semantic_Orientations_in_Economic_Texts).  
2. The dataset released as part of the open challenge for Financial Opinion Mining and Question Answering ([FiQA](https://sites.google.com/view/fiqa/home)) in 2018.
3. The News Statements and Headlines dataset released for Task 5: [Fine-Grained Sentiment Analysis on Financial Microblogs and News](https://www.aclweb.org/anthology/S17-2089/)" in SemEval 2017.

The Financial Phrasebank datasets are given discrete labels of "positive", "negative" and "neutral" for each given sentence. The differences between the 4 sub-datasets depend on the percentage of experts that agree with the sentiment of the financial news sentence, ranging from where 50%, 66%, 75% and 100% consensus. Some examples of labels assigned to news sentences are as follows:

#### Sentences_50Agree.txt
| Headline | Sentiment |
|:-:|:-:|
| According to the company 's updated strategy for the years 2009-2012 , Basware targets a long-term net sales growth in the range of 20 % -40 % with an operating profit margin of 10 % -20 % of net sales . | positive |
| The international electronic industry company Elcoteq has laid off tens of employees from its Tallinn facility ; contrary to earlier layoffs the company contracted the ranks of its office workers , the daily Postimees reported . | negative |
| According to Gran , the company has no plans to move all production to Russia , although that is where the company is growing . | neutral |

#### Sentences_66Agree.txt
| Headline | Sentiment |
|:-:|:-:|
| With the new production plant the company would increase its capacity to meet the expected increase in demand and would improve the use of raw materials and therefore increase the production profitability . | positive |
| Compared with the FTSE 100 index , which rose 94.9 points ( or 1.6 % ) on the day , this was a relative price change of -0.4 % . | negative |
| In Sweden , Gallerix accumulated SEK denominated sales were down 1 % and EUR denominated sales were up 11 % . | neutral |

#### Sentences_75Agree.txt
| Headline | Sentiment |
|:-:|:-:|
| In the third quarter of 2010 , net sales increased by 5.2 % to EUR 205.5 mn , and operating profit by 34.9 % to EUR 23.5 mn . | positive |
| Jan. 6 -- Ford is struggling in the face of slowing truck and SUV sales and a surfeit of up-to-date , gotta-have cars . | negative |
| At the same time , the market for automated liquid handling devices is already larger than that for pipettes , according to Biohit . | neutral |

#### Sentences_AllAgree.txt
| Headline | Sentiment |
|:-:|:-:|
| Operating profit totalled EUR 21.1 mn , up from EUR 18.6 mn in 2007 , representing 9.7 % of net sales . | positive |
| Pharmaceuticals group Orion Corp reported a fall in its third-quarter earnings that were hit by larger expenditures on R&D and marketing . | negative |
| Simultaneously , his responsibility area is extended from legal affairs to cover also mergers and acquisitions . | neutral |

The summary of FinBERT's performance for each of the sub-datasets are as follows:

#### Sentences_75Agree.txt
| |  Precision | Recall | F1-Score |
|:-:|:-:|:-:|:-:|
| Negative | 0.86 | 0.98 | 0.92 |
| Neutral | 0.99 | 0.93 | 0.96 |
| Positive | 0.90 | 0.97 | 0.93 |

#### Sentences_AllAgree.txt
| |  Precision | Recall | F1-Score |
|:-:|:-:|:-:|:-:|
| Negative | 0.91 | 0.98 | 0.94 |
| Neutral | 1.00 | 0.97 | 0.98 |
| Positive | 0.95 | 0.98 | 0.96 |

FinBERT sentiment analysis model is now available on Hugging Face model hub. You can get the model [here](https://huggingface.co/ProsusAI/finbert). 

FinBERT is a pre-trained NLP model to analyze sentiment of financial text. It is built by further training
 the [BERT](https://arxiv.org/pdf/1810.04805.pdf) language model in the finance domain, using a large financial corpus and thereby fine-tuning
  it for financial sentiment classification. For the details, please see 
  [FinBERT: Financial Sentiment Analysis with Pre-trained Language Models](https://arxiv.org/pdf/1908.10063.pdf).

**Important Note:** 
FinBERT implementation relies on Hugging Face's `pytorch_pretrained_bert` library and their implementation of BERT for sequence classification tasks. `pytorch_pretrained_bert` is an earlier version of the [`transformers`](https://github.com/huggingface/transformers) library. It is on the top of our priority to migrate the code for FinBERT to `transformers` in the near future.

## Installing
 Install the dependencies by creating the Conda environment `finbert` from the given `environment.yml` file and
 activating it.
```bash
conda env create -f environment.yml
conda activate finbert
```

## Models
FinBERT sentiment analysis model is now available on Hugging Face model hub. You can get the model [here](https://huggingface.co/ProsusAI/finbert). 

Or, you can download the models from the links below:
* [Language model trained on TRC2](https://prosus-public.s3-eu-west-1.amazonaws.com/finbert/language-model/pytorch_model.bin)
* [Sentiment analysis model trained on Financial PhraseBank](https://prosus-public.s3-eu-west-1.amazonaws.com/finbert/finbert-sentiment/pytorch_model.bin)

For both of these model, the workflow should be like this:
* Create a directory for the model. For example: `models/sentiment/<model directory name>`
* Download the model and put it into the directory you just created.
* Put a copy of `config.json` in this same directory. 
* Call the model with `.from_pretrained(<model directory name>)`

## Datasets
There are two datasets used for FinBERT. The language model further training is done on a subset of Reuters TRC2 
dataset. This dataset is not public, but researchers can apply for access 
[here](https://trec.nist.gov/data/reuters/reuters.html).

For the sentiment analysis, we used Financial PhraseBank from [Malo et al. (2014)](https://www.researchgate.net/publication/251231107_Good_Debt_or_Bad_Debt_Detecting_Semantic_Orientations_in_Economic_Texts).
 The dataset can be downloaded from this [link](https://www.researchgate.net/profile/Pekka_Malo/publication/251231364_FinancialPhraseBank-v10/data/0c96051eee4fb1d56e000000/FinancialPhraseBank-v10.zip?origin=publication_list).
 If you want to train the model on the same dataset, after downloading it, you should create three files under the 
 `data/sentiment_data` folder as `train.csv`, `validation.csv`, `test.csv`. 
To create these files, do the following steps:
- Download the Financial PhraseBank from the above link.
- Get the path of `Sentences_50Agree.txt` file in the `FinancialPhraseBank-v1.0` zip.
- Run the [datasets script](scripts/datasets.py):
```python scripts/datasets.py --data_path <path to Sentences_50Agree.txt>```

## Training the model
Training is done in `finbert_training.ipynb` notebook. The trained model will
 be saved to `models/classifier_model/finbert-sentiment`. You can find the training parameters in the notebook as follows:
```python
config = Config(   data_dir=cl_data_path,
                   bert_model=bertmodel,
                   num_train_epochs=4.0,
                   model_dir=cl_path,
                   max_seq_length = 64,
                   train_batch_size = 32,
                   learning_rate = 2e-5,
                   output_mode='classification',
                   warm_up_proportion=0.2,
                   local_rank=-1,
                   discriminate=True,
                   gradual_unfreeze=True )
```
The last two parameters `discriminate` and `gradual_unfreeze` determine whether to apply the corresponding technique 
against catastrophic forgetting.

## Getting predictions
We provide a script to quickly get sentiment predictions using FinBERT. Given a .txt file, `predict.py` produces a .csv file including the sentences in the text, corresponding softmax probabilities for three labels, actual prediction and sentiment score (which is calculated with: probability of positive - probability of negative).

Here's an example with the provided example text: `test.txt`. From the command line, simply run:
```bash
python3 predict.py --output_dir output/semeval2017_output --input_dir input/semeval_2017 --model_dir models/sentiment/
```

Where:
- `input_dir`: folder containing .csv files which you want to process
- `model_dir`: folder containing models which you want to use to process .csv files
- `output_dir`: folder containing output sentiment classification and sentiment score

## Disclaimer
This is not an official Prosus product. It is the outcome of an intern research project in Prosus AI team.
### About Prosus 
Prosus is a global consumer internet group and one of the largest technology investors in the world. Operating and
 investing globally in markets with long-term growth potential, Prosus builds leading consumer internet companies that empower people and enrich communities.
For more information, please visit [www.prosus.com](www.prosus.com).

## Contact information
Please contact Dogu Araci `dogu.araci[at]naspers[dot]com` and Zulkuf Genc `zulkuf.genc[at]naspers[dot]com` about
 any FinBERT related issues and questions.
