import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

@st.cache
def get_metadata():
    # load data of all grocery items
    item_df = pd.read_csv('./data/agg_df_nutrient_chart_ver20210126.csv')
    return item_df

def get_source():
    # load additional information
    source_df= pd.read_csv('./data/agg_price_image_list.csv')
    return source_df


def user_input_features():
    # create a sidebar for a user input & a dataframe
    market=st.selectbox("Choose your supermarket", ['Morrison','Tesco','M&S','Sainsburys','Waitrose'])
    energy_kcal=st.slider('Targeted Daily Intake (kcal)', min_value=1000, max_value=6000, step=10)
    days=st.slider('Shopping Volume (Days)  (e.g. 6 days)', min_value=3, max_value=7, step=1)

    data={'Target': [market, energy_kcal, days]}
    user_if=pd.DataFrame(data, index=['store','kcal','days'])
    return user_if

def user_input_processing(kcal, days):
    # calculate a whole nutrition values according to a user input
    rdi=[2000, 32, 10, 300, 25, 28, 50, 3]
    ratio=int(kcal)/rdi[0]
    target=[int(ele*ratio*days) for ele in rdi]
    order_title=['Energy(kcal)','Fat(g)','Saturated_fat(g)','Carbohydrate(g)','of_which_sugars(g)','Fibre(g)','Protein(g)','Salt(g)']
    target_nutrition=pd.DataFrame(np.array(target), index=order_title, columns=['Total Values'])
    
    return target_nutrition

def recommendation(item_df, target_vector):
    
    norm_df=item_df.iloc[:,1:]/target_vector
    num_item=int(target_vector[0]/item_df['Energy(kcal)'].mean())
    output_list=[]
    mix_n=0

    while len(output_list)<2:
        agg_i=np.random.choice(len(item_df),num_item)
        codes=np.sum(norm_df.to_numpy()[agg_i], axis=0, keepdims=False)

        observation = np.repeat(1, len(target_vector))
        diff = codes - observation
        dist = np.sqrt(np.sum(diff**2,axis=-1))
        nearest = np.argmin(dist)

        if dist <0.5:
            mix_n=mix_n+1
            shopping_list=item_df.loc[agg_i.tolist(),'item name'].tolist()
            data={'Shopping mix': shopping_list, 'Content':[codes.tolist()]*len(shopping_list), 'Similarity': [str(dist)]*len(shopping_list)}
            df=pd.DataFrame(data)
            output_list.append(df)

        else:
            pass
    
    return output_list


def create_mix_list(output_list):
    mix_list=output_list.loc[:,'Shopping mix'].copy()
    
    return mix_list


def create_img_url_list(name_df, source_df):
    img_url_list=[]
    for name in name_df.tolist():
        if len(source_df['image_url'][source_df['item_name'].str.contains(name, case=False)])==1:
            url=source_df['image_url'][source_df['item_name'].str.contains(name, case=False)].values[0]
            img_url_list.append(url)
        else:
            rename=name.rsplit(" ", 1)[0]
            if len(source_df['image_url'][source_df['item_name'].str.contains(rename, case=False)])==1:
                url=source_df['image_url'][source_df['item_name'].str.contains(rename, case=False)].values[0]
                img_url_list.append(url)
            else:
                rerename=rename.rsplit(" ", 1)[0]
                if len(source_df['image_url'][source_df['item_name'].str.contains(rerename, case=False)])==1:
                    url=source_df['image_url'][source_df['item_name'].str.contains(rerename, case=False)].values[0]
                    img_url_list.append(url)
                else:
                    rererename=rename.rsplit(" ", 1)[0]
                    if len(source_df['image_url'][source_df['item_name'].str.contains(rererename, case=False)])==1:
                        url=source_df['image_url'][source_df['item_name'].str.contains(rererename, case=False)].values[0]
                        img_url_list.append(url)
                    else:
                        pass
    return img_url_list




def create_barplot(content_vector1, content_vector2):
    con1=[round(float(ele)*100,2) for ele in content_vector1]
    con2=[round(float(ele)*100,2) for ele in content_vector2]
    fig, ((ax1, ax2))=plt.subplots(2,1, figsize=(5,10))
    fig.subplots_adjust(hspace = 0.5)
    x=['Energy','Fat','Saturated_fat','Carbohydrate','of_which_sugars','Fibre','Protein','Salt']
    y0=np.repeat(100,8)
    y1=con1
    y2=con2
    # plot the 2 sets
    ax1.plot(x,y0, label='Target Level', lw=3, alpha=0.5)
    ax1.bar(x,y1, label='Shopping Mix 1', color='green')
    ax2.plot(x,y0, label='Target Level', lw=3, alpha=0.5)
    ax2.bar(x,y2, label='Shopping Mix 2', color='navy')
    ax1.set_xticklabels(x, rotation=45, ha='right')
    ax2.set_xticklabels(x, rotation=45, ha='right')
    ax1.set_title("Shopping Mix 1: Nutrient Content (%)")
    ax2.set_title("Shopping Mix 2: Nutrient Content (%)")
    ax1.set_ylim([0, 200])
    ax2.set_ylim([0, 200])
    ax1.set_ylabel('Nutrient Content/Target Amount (%)')
    ax2.set_ylabel('Nutrient Content/Target Amount (%)')
    ax1.legend(loc=2)
    ax2.legend(loc=2)
    
    return st.pyplot(fig)


def displayMe():
    """
    Demo
    """
    import sys
    thismodule = sys.modules[__name__]

    st.write("Your Information")
    user_if
    
    st.write("##")
    
    my_expander = st.beta_expander("See detail by typical values", expanded=False)
    with my_expander:
        st.write(target_nutrition)
        
    st.write("##")
    
    st.write("Shopping Mix 1")
    mix_list1
    
    st.write("##")
    
    st.image(img_url_list1)
    
    st.write("##")
    
    st.write("Shopping Mix 2")
    mix_list2
    
    st.write("##")
    
    st.image(img_url_list2)
    
    st.write("##")
    
    st.write("Shopping Mixes: Nutrition Target Achievement %")
    create_barplot(output_list[0].loc[0,'Content'], output_list[1].loc[0,'Content'])


    
st.title("Make your grocery shopping more balanced!")
st.header("Tell Me About Your Target Calories")
# excute functions
item_df=get_metadata()
source_df=get_source()
user_if=user_input_features()
target_nutrition=user_input_processing(user_if.iloc[1,0], user_if.iloc[2,0])
output_list=recommendation(item_df, target_nutrition.iloc[:,0])
mix_list1=create_mix_list(output_list[0])
mix_list2=create_mix_list(output_list[1])
img_url_list1=create_img_url_list(mix_list1, source_df)
img_url_list2=create_img_url_list(mix_list2, source_df)
st.header("Grocery Items According To Your Target")
displayMe()


