PATH_ROOT="." # change this path to your project directory path

PATH_HURRICANE = PATH_ROOT+"/data/hurricane/datasets_raw/"
PATH_HURRICANE_PROCESSED = PATH_ROOT+"/data/hurricane/datasets_binary/"
PATH_GOEMOTIONS = PATH_ROOT+"/data/goemotions/data/"
PATH_MAPPINGS = PATH_ROOT+"/data/emo-mapping/"
PATH_PROCESSED = PATH_ROOT+"/data/processed-emotions/"
PATH_MODELS = PATH_ROOT+"/trained_models/"
PATH_SAVE_EMOTION = PATH_ROOT+"/res/emotion_classification/"
PATH_SAVE_EMOTION_BINARY = PATH_ROOT+"/res/emotion_classification_binary/"
PATH_SAVE_2014_EMOTION_BINARY = PATH_ROOT+"/res/emotion_classification_2014_binary/"

EMO_EKMAN = ["disgust", "fear", "anger", "sadness", "surprise", "joy"]
EMO_DATA_AVAILABLE = ["go","hurricane","blm"]

BERT_MAX_LENGTH = 60
BERT_DIR = PATH_ROOT+"/transformer-models/"
PATH_BERT_BLM = BERT_DIR+"bert_blm_from-pre-trained/best_model"
PATH_BERT_BLM_ONLY = BERT_DIR+"bert_blm/best_model"
PATH_BERT_HUR = BERT_DIR+"bert_hurricane_ext_from-pre-trained/best_model"
PATH_BERT_HUR_ONLY = BERT_DIR+"bert_hurricane_ext/best_model"

PATH_SUMMARY = PATH_ROOT+"/model_summary/"