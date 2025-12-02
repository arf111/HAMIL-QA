import copy
from tqdm import tqdm
import json

def get_qc_scores(data_files, qc_dict_json_path):
    with open(qc_dict_json_path, 'r') as f:
        qc_dict = json.load(f)

    qc_scores = []
    qc_dict_scores = {}
    labeled_data_files = []
    data_files_with_labels = []

    for data_file in data_files:
        data_file_name = data_file.split("/")[-1]

        if qc_dict[data_file_name]["label"] != 0:

            labeled_data_files.append(data_file)
            data_files_with_labels.append(data_file)

            qc_scores.append(qc_dict[data_file_name]["label"]["quality_for_fibrosis_assessment"])
            
            if data_file_name not in qc_dict_scores:
                qc_dict_scores[data_file_name] = {"label": {}}

            qc_dict_scores[data_file_name]["label"]["quality_for_fibrosis_assessment"] = 0.0 if qc_dict[data_file_name]["label"]["quality_for_fibrosis_assessment"] <= 2.0 else 1.0
                                
    return qc_scores, labeled_data_files, qc_dict_scores

def get_patient_records_monai(filenames, data_category='train', qc_dict_json=None):
    patients = []

    for subject_id in tqdm(range(len(filenames)), desc=f'Loading {data_category} data', unit='subject'):
        subject_name = filenames[subject_id].split('/')[-1]
        image_name = filenames[subject_id] + '/data.nrrd'
        la_label_name = filenames[subject_id] + '/shrinkwrap.nrrd'
        qc_labels = copy.deepcopy(qc_dict_json[subject_name]['label'])
        
        if qc_labels == 0:
            raise ValueError(f"Quality control label is 0 for {subject_name}")
    
        data = {'image': image_name, 
                'la_label': la_label_name,
                'labels': qc_labels,
                'p_id': subject_name,
                }

        patients.append(data)

    return patients