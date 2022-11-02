import argparse
import pickle
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from preprocessing import prepare_test_data

def make_prediction(model, dataset, selected_variables, result_path):
    test = dataset[selected_variables]
    idx = dataset[['transcript_id', 'position']]
    print('Making Prediction')

    predicted_proba = model.predict_proba(test)[::,1]
    df = pd.concat([idx, pd.DataFrame(predicted_proba)], axis = 1)
    df.columns = ['transcript_id', 'transcript_position', 'score']
    print(f'Saving Model Prediction Result as {result_path}')
    df.to_csv(result_path, index = False)


def main():
    parser = argparse.ArgumentParser(description='Testing Script')
    parser.add_argument('--json_path', type=str, default='../data/data.json', help='Specify the path to json file here', required=True)
    parser.add_argument('--data_out_path', type=str, default='../data/parsed_prediction_data.csv', help='Specify the path to save processed test data', required=True)
    parser.add_argument('--model_path', type=str, default='../model/rf_model.sav', help='Specify the path to obtain trained model', required=False)
    parser.add_argument('--prediction_out_path', type=str, default='../output/predicted_labels.csv', help='Specify the path to save predicted labels', required=True)
    parser.add_argument('--selected_variables_path', type=str, default='../data/selected_variables.txt', help='Specify the path to obtain selected variables', required=False)
    args = vars(parser.parse_args())

    df_processed = prepare_test_data(json_path = args['json_path'], result_path = args['data_out_path'])

    print('Loading trained model')
    model = pickle.load(open(args['model_path'], 'rb'))

    print('Loading selected variables')
    selected_variables = np.loadtxt(args['selected_variables_path'], dtype=str, comments="#", delimiter=",", unpack=False)

    make_prediction(model = model, dataset = df_processed, selected_variables = selected_variables, result_path = args['prediction_out_path'])


if __name__ == '__main__':
    main()