from tensorflow import keras
from keras_preprocessing.sequence import pad_sequences
import numpy as np
import pandas as pd
import json
from lstm_kmers import train_lstm_model

###########################################
# Preparing LSTM model for sequence value #
###########################################
BATCH_SIZE=500

def kmers_funct(seq, size=5):
    return [seq[x:x+size].lower() for x in range(len(seq) - size + 1)]

def letter_to_num(letter):
    kmers_sequence = kmers_funct(letter, size=5)
    _kmers = ['aaaac', 'aaaca', 'aaacc', 'aaact', 'aacaa', 'aacac', 'aacag',
       'aacat', 'aacca', 'aaccc', 'aaccg', 'aacct', 'aacta', 'aactc',
       'aactg', 'aactt', 'aagac', 'agaac', 'agaca', 'agacc', 'agact',
       'aggac', 'ataac', 'atgac', 'caaac', 'cagac', 'cgaac', 'cggac',
       'ctaac', 'ctgac', 'gaaac', 'gaaca', 'gaacc', 'gaact', 'gacaa',
       'gacac', 'gacag', 'gacat', 'gacca', 'gaccc', 'gaccg', 'gacct',
       'gacta', 'gactc', 'gactg', 'gactt', 'gagac', 'ggaac', 'ggaca',
       'ggacc', 'ggact', 'gggac', 'gtaac', 'gtgac', 'taaac', 'taaca',
       'taacc', 'taact', 'tagac', 'tgaac', 'tgaca', 'tgacc', 'tgact',
       'tggac', 'ttaac', 'ttgac']
    _kmers_dict = dict([(_kmer,i) for i, _kmer in enumerate(_kmers)])
    return [_kmers_dict.get(i, None) for i in kmers_sequence]

def pred_train_sequence_value(df):
    sequence_num = pad_sequences(df['sequence'].apply(letter_to_num), maxlen=3)
    labels = np.array(df['label'].values)

    print('Loading lstm model')
    # load fitted lstm model if exist, else train lstm model and saved in folder 'model'
    try:
        lstm_model = keras.models.load_model("../model/lstm_kmers_sequence_model")
    except:
        print('LSTM model is not trained. Starting the training process')
        lstm_model = train_lstm_model(sequence_num, labels)

    return lstm_model.predict(sequence_num, batch_size=BATCH_SIZE)

def pred_sequence_value(df):
    sequence_num = pad_sequences(df['sequence'].apply(letter_to_num), maxlen=3)

    print('Loading lstm model')
    try:
        lstm_model = keras.models.load_model("../model/lstm_kmers_sequence_model")
        return lstm_model.predict(sequence_num, batch_size=BATCH_SIZE)
    except:
        raise ValueError('LSTM model is not trained. Starting the training process by running training script')
        
################
# Parsing JSON #
################

def parse_data(data):
    ls = []
    temp = data 
    first = list(temp.items())[0]
    trans_name = first[0]

    second = list(first[1].items())[0]
    pos = second[0]

    third = list(second[1].items())[0]
    seq = third[0]
    
    data = third[1]
    
    avg = np.mean(data, axis = 0)
    med = np.median(data, axis = 0)
    minn = np.min(data, axis = 0)
    maxx = np.max(data, axis = 0)
    std = np.std(data, axis = 0)
    
    ls += [trans_name, pos, seq]
    ls += list(avg)+ list(med) + list(minn) + list(maxx)+ list(std)
    
    return ls
    

def load_json(json_path) -> pd.DataFrame:
    print('Parsing json data')
    genome = []
    for line in open(json_path, 'r'):
        genome.append(json.loads(line))

    df = pd.DataFrame()
    result = pd.DataFrame(list(map(lambda x: parse_data(x), genome)))

    colname = ['transcript', 'position', 'sequence']
    for level in ['prev', 'curr', 'next']:
        for name in ["dwellingtime", "std", "meancurrent"]:
            for stats in ['avg', 'med', 'min', 'max', 'std']:
                colname.append(level+'_'+name+'_'+stats)
    result.columns = colname
    result['position'] = result['position'].astype(int)
    return result

######################
# Prepare Train Data #
######################

def load_label(info_path) -> pd.DataFrame:
    print('Reading label data.')
    label = pd.read_csv(info_path)
    return label

def prepare_train_data(json_path, info_path, result_path) -> pd.DataFrame:
    df_json_data = load_json(json_path)
    label_data = load_label(info_path)
    
    df = df_json_data.merge(label_data, left_on = ['transcript', 'position'], right_on = ['transcript_id', 'transcript_position'], how = 'inner').drop(['transcript', 'transcript_position'], axis = 1)
    df = df[['gene_id', 'transcript_id']+[col for col in df_json_data.columns if col != 'transcript']+['label']]

    print('Getting sequence value from LSTM model.')
    sequence_value = pred_train_sequence_value(df)
    df['sequence_value'] = sequence_value

    print(f'Train data saved as {result_path}')
    df.to_csv(result_path, index=False)
    return df

######################
# Prepare Test Data #
######################
def prepare_test_data(json_path, result_path) -> pd.DataFrame:
    df_json_data = load_json(json_path)

    sequence_value = pred_sequence_value(df_json_data)
    df_json_data['sequence_value'] = sequence_value
    df_json_data.rename(columns={'transcript': 'transcript_id'}, inplace=True)

    print(f'Parsed prediction data saved as {result_path}')
    df_json_data.to_csv(result_path, index=False)
    return df_json_data
