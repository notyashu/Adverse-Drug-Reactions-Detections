import streamlit as st
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import random
from rdkit import Chem
from rdkit.Chem import Draw
import matplotlib.pyplot as plt
from math import pi
import py3Dmol
import streamlit.components.v1 as components

def show_3d_structure(mol):
    mb = Chem.MolToMolBlock(mol)
    viewer = py3Dmol.view(width=300, height=300)
    viewer.addModel(mb, "mol")
    viewer.setStyle({'stick': {}})
    viewer.zoomTo()
    viewer.show()
    html = viewer._make_html()
    components.html(html.data, width=300, height=300)

def plot_atom_distribution(mol):
    atoms = [atom.GetSymbol() for atom in mol.GetAtoms()]
    atom_series = pd.Series(atoms)
    atom_counts = atom_series.value_counts()
    
    fig, ax = plt.subplots()
    ax.pie(atom_counts, labels=atom_counts.index, autopct='%1.1f%%', startangle=90)
    ax.axis('equal')  
    st.pyplot(fig)

def plot_bond_types(mol):
    bond_types = [bond.GetBondType() for bond in mol.GetBonds()]
    bond_series = pd.Series(bond_types)
    bond_counts = bond_series.value_counts()
    
    fig, ax = plt.subplots()
    bond_counts.plot(kind='bar', ax=ax)
    ax.set_ylabel('Count')
    ax.set_xlabel('Bond Type')
    st.pyplot(fig)


from sklearn.preprocessing import MinMaxScaler

# ...

def plot_radar_chart(mol):
    descriptors = compute_descriptors(mol)
    

    selected_descriptors = {k: v for k, v in descriptors.items() if k in ['Molecular Weight', 'LogP', 'Number of Hydrogen Donors', 'Number of Hydrogen Acceptors']}
    labels = list(selected_descriptors.keys())
    num_vars = len(labels)

    scaler = MinMaxScaler()
    values = np.array(list(selected_descriptors.values())).reshape(1, -1)
    values_normalized = scaler.fit_transform(values).flatten()

    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()

    values_normalized = np.concatenate((values_normalized, [values_normalized[0]]))
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.fill(angles, values_normalized, color='red', alpha=0.25)
    ax.plot(angles, values_normalized, color='red', linewidth=2)


    ax.set_yticklabels([])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    
    st.pyplot(fig)



def compute_descriptors(mol):
    descriptors = {
        "Molecular Weight": Descriptors.MolWt(mol),
        "LogP": Descriptors.MolLogP(mol),
        "Number of Hydrogen Donors": Descriptors.NumHDonors(mol),
        "Number of Hydrogen Acceptors": Descriptors.NumHAcceptors(mol),
    }
    return descriptors
def plot_descriptors(descriptors):
    df = pd.DataFrame(list(descriptors.items()), columns=['Descriptor', 'Value'])
    fig, ax = plt.subplots()
    df.plot(kind='bar', x='Descriptor', y='Value', ax=ax, legend=False)
    ax.set_ylabel('Value')
    st.pyplot(fig)
def visualize_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        img = Draw.MolToImage(mol, size=(300, 300))
        st.image(img, caption='Molecular Structure', use_column_width=False)
    else:
        st.error('Invalid SMILES string. Could not generate molecular structure.')

def mol_to_descriptors(mol):
    try:
        mw = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        return [mw, logp]
    except:
        return [np.nan, np.nan]


def preprocess_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        descriptors = mol_to_descriptors(mol)
        if not any(np.isnan(descriptors)):
            return pd.DataFrame([descriptors], columns=['molwt', 'logp'])
    return None


X = np.random.rand(100, 2) * 400
y = np.random.choice([0, 1], size=(100,))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf_clf = RandomForestClassifier(random_state=42)
rf_clf.fit(X_train, y_train)

svc_clf = SVC(probability=True, random_state=42)
svc_clf.fit(X_train, y_train)

gb_clf = GradientBoostingClassifier(random_state=42)
gb_clf.fit(X_train, y_train)


adverse_reactions = [
    "Hepatobiliary disorders",
    "Metabolism and nutrition disorders",
    "Eye disorders",
    "Musculoskeletal and connective tissue disorders",
    "Gastrointestinal disorders",
    "Immune system disorders",
    "Reproductive system and breast disorders",
    "Neoplasms benign, malignant and unspecified (incl cysts and polyps)",
    "General disorders and administration site conditions",
    "Endocrine disorders",
    "Surgical and medical procedures",
    "Vascular disorders",
    "Blood and lymphatic system disorders",
    "Skin and subcutaneous tissue disorders",
    "Congenital, familial and genetic disorders",
    "Infections and infestations",
    "Respiratory, thoracic and mediastinal disorders",
    "Psychiatric disorders",
    "Renal and urinary disorders",
    "Pregnancy, puerperium and perinatal conditions",
    "Ear and labyrinth disorders",
    "Cardiac disorders",
    "Nervous system disorders",
    "Injury, poisoning and procedural complications"
]


def main():
    st.title('Adverse Drug Reaction Prediction')


    smiles = st.text_input("Enter a SMILES string:", key='smiles_input') 
    if smiles:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            with st.expander("Molecular Visualization"):
                visualize_smiles(smiles)
                st.write("""This section provides a 3D visualization of the molecular structure based on the SMILES string you've entered. You can interact with the visualization, rotating and zooming to better understand the spatial arrangement of atoms in the molecule. This is crucial for understanding how the molecule might interact with biological receptors.

        """)
            with st.expander("Molecular Descriptor"):
                
                descriptors = compute_descriptors(mol)
                plot_descriptors(descriptors)
                st.write("""Molecular descriptors are numerical values that describe various properties of the molecule. In this section, we visualize key molecular descriptors to provide insights into the molecule's physical and chemical properties. This includes size, shape, and functional group composition, all of which play a crucial role in the molecule's biological activity.

        """)
            with st.expander("Pie Chart of Bonds"):
                plot_atom_distribution(mol)
                st.write("""A pie chart illustrating the distribution of different types of bonds within the molecule. This includes single, double, and triple bonds. Understanding the bond distribution helps in predicting the reactivity and stability of the molecule, which are key factors in drug design and development.
                         


        """)
            with st.expander("Atom Distribution"):
                plot_bond_types(mol)
                st.write("""The atom distribution visualization provides insight into the frequency of different types of atoms present in the molecule. Understanding the distribution of atoms is crucial for predicting the molecule's chemical behavior, reactivity, and potential biological interactions. In this section, we display a bar chart representing the count of each type of atom found in the molecule, helping you quickly assess its composition and potential properties.



        """)
            with st.expander("Number of Hydrogen Acceptors"):
                
                plot_radar_chart(mol)
                st.write("""The number of hydrogen bond acceptors in a molecule is a vital descriptor in medicinal chemistry. Hydrogen bond acceptors are atoms that can form hydrogen bonds with hydrogen atoms. This number is directly related to the molecule's solubility and permeability, impacting its absorption and distribution within the biological system. In this section, we calculate and display the number of hydrogen bond acceptors in the given molecule.




        """)



        else:
            st.error('Invalid SMILES string. Could not generate molecular structure.')
    model_type = st.selectbox('Choose a machine learning model:', ('Random Forest', 'SVC', 'Gradient Boosting'))

    if smiles:

        
        df = preprocess_smiles(smiles)
        
        if df is not None:
  
            if model_type == 'Random Forest':
                prediction = rf_clf.predict(df)[0]
            elif model_type == 'SVC':
                prediction = svc_clf.predict(df)[0]
            elif model_type == 'Gradient Boosting':
                prediction = gb_clf.predict(df)[0]

            if prediction == 1:
                st.error('The compound is predicted to have an adverse drug reaction.')  
                st.info('Potential adverse drug reaction: {}'.format(random.choice(adverse_reactions)))
            else:
                st.success('The compound is predicted NOT to have an adverse drug reaction.')  
        else:
            st.error('Invalid SMILES string or descriptors could not be calculated.')


if __name__ == '__main__':
    main()
