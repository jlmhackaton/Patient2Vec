import numpy as np
import pandas as pd

# create the output file
# read the addmission file
addmissions_file = pd.read_csv('ADMISSIONS_demo.csv')
addmissions_number = np.size(addmissions_file, 0)
HADM_ID = addmissions_file.get('HADM_ID')


# red the microbiologicalevents file
microbiologicalevents_file = pd.read_csv('MICROBIOLOGYEVENTS_demo.csv', usecols=['HADM_ID', 'INTERPRETATION'])
# microbiologicalevents_file = microbiologicalevents_file.sort_values(by=('HADM_ID'))

INTERPRETATION = microbiologicalevents_file.get('INTERPRETATION')

dignostics = []
for ind1 in addmissions_file.get('HADM_ID'):
    all_microbiologicalevents_for_one_subject = microbiologicalevents_file.index[microbiologicalevents_file['HADM_ID']
                                                                                 == ind1].tolist()
    temp_list = INTERPRETATION[all_microbiologicalevents_for_one_subject]
    if any(temp_list == 'S') or any(temp_list == 'I'):
        dignostics.append(1)
    else:
        dignostics.append(0)
output_dataframe = pd.DataFrame(data={'HADM_ID': np.array(HADM_ID), 'INTERPRETATION': np.array(dignostics)})
output_dataframe.to_csv('Output.csv')
