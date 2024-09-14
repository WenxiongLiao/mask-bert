import pandas as  pd
import os

# from active_learning.AL import AL_detect
from build_model.build_ml import collate_function,tokenize_function



def data_prep(source_save_path,dest_save_path,labels):
    data_df = pd.read_csv(source_save_path,sep="\t")
    select_idx = [True if label in labels else False for label in data_df['labels'].values]
    data_df = data_df[select_idx]
    data_df = data_df.reset_index(drop=True)
    data_df.to_csv(dest_save_path,index = False)

#AG news and dbpedia
def AG_dbpedia_preprocessing(source_save_path,dest_save_path,labels):
    data_df = pd.read_csv(source_save_path,sep="\t")
    data_df['text'] = data_df['Title'] + ' . ' +  data_df['Description']
    data_df['labels'] = data_df['Class Index']
    data_df = data_df[['labels','text']]
    select_idx = [True if label in labels else False for label in data_df['labels'].values]
    data_df = data_df[select_idx]
    data_df = data_df.reset_index(drop=True)
    data_df.to_csv(dest_save_path,index = False)


def build_few_shot_samples(novel_data_path,novel_labels,K_shot,novel_few_shot_path,random_state,is_AL = False):

    if is_AL:
        # few_shot_df = AL_detect(model_checkpoint, device,novel_data_path,novel_labels,K_shot,random_state,collate_function,tokenize_function )
        # novel_few_shot_path = novel_few_shot_path[:-4] + '_' + str(K_shot) + '_' + str(random_state) + novel_few_shot_path[-4:]
        # few_shot_df =  pd.read_csv(novel_few_shot_path)
        # We have obtained activate learning samples with coreset
        pass

    else:    
        data_df = pd.read_csv(novel_data_path)
        few_shot_df = None
        for label in novel_labels:
            tmp = pd.DataFrame(data_df[data_df.labels == label])
            tmp = tmp.sample(n=K_shot,random_state = random_state)
            if type(few_shot_df) == type(None):
                few_shot_df = tmp
            else:
                few_shot_df = pd.concat([few_shot_df,tmp], ignore_index=True)
        few_shot_df = few_shot_df.reset_index(drop=True)
        
        save_dir = os.path.dirname(novel_few_shot_path)
        if os.path.exists(save_dir) == False:
            os.makedirs(save_dir)
        few_shot_df.to_csv(novel_few_shot_path,index = False)

