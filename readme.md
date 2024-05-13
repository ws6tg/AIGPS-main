# Adaptive individualized gene pair signatures distinguishing melanoma and predicting response to immune checkpoint blockade
## introduction
Distinguishing among similar cancer types and predicting responses to immune checkpoint blockade (ICB) therapies are critical to reduce mortality and cost in cancer diagnosis and treatment. Several gene expression signatures have been developed for the multiclass classification and ICB response prediction, but they suffered from batch effect and technical variations when generalizing to other cohorts in clinical practice. Although rank-based method can mitigate the technical variations, it ignores the quantitative alteration caused by cancer. Therefore, we propose an adaptive quantification of rank-based method and extract Adaptive Individualized Gene Pair Signatures (AIGPS) to classify multi-class skin cancer and predict ICB in melanoma. AIGPS method utilizes adaptive quantification of the difference between each gene pair and identifies significant reversed gene pairs in each skin cancer types and in responses to ICB across single-cell and bulk RNA sequencing data with machine learning algorithms. 
![](./overview.png 'Overview')

## Prerequisite
- **python3.8**
- **Python packages:** requirements.txt

## immunotherapy response prediction
### datasets
- **discovery cohort:**  **scRNA** [GSE120575](data/Pred/sc/GSE120575.h5ad)
- **training cohorts:**  **bulk** 
    - **Van** [Van_TPM.csv](data/Pred/bulk/Van_TPM.csv) [Van_anno.csv](data/Pred/bulk/Van_anno.csv)
    - **Riaz** [Riaz_TPM.csv](data/Pred/bulk/Riaz_TPM.csv) [Riaz_anno.csv](data/Pred/bulk/Riaz_anno.csv)
- **validation cohorts:**  **bulk** 
    - **MGH** [MGH_anno.csv](data/Pred/bulk/MGH_anno.csv) [MGH_TPM.csv](data/Pred/bulk/MGH_TPM.csv) 
    - **Gide** [Gide_anno.csv](data/Pred/bulk/Gide_anno.csv) [Gide_TPM.csv](data/Pred/bulk/Gide_TPM.csv) 
    - **Hugo** [Hugo_anno.csv](data/Pred/bulk/Hugo_anno.csv) [Hugo_TPM.csv](data/Pred/bulk/Hugo_TPM.csv) 
    - **Lee** [Lee_anno.csv](data/Pred/bulk/Lee_anno.csv) [Lee_TPM.csv](data/Pred/bulk/Lee_TPM.csv)

    **Table 1: Datasets information in immunotherapy response prediction**

    <table><tr><th colspan="1" rowspan="2"><b>Cohort</b></th><th colspan="1" rowspan="2"><b>Technology</b></th><th colspan="3"><b>Pre-Treatments Samples (Cells)</b></th><th colspan="1" rowspan="1"><b>Treatment</b></th><th colspan="1" rowspan="1"><b>Reference</b></th></tr>
    <tr><td colspan="1"><b>Response</b></td><td colspan="1"><b>Non-Response</b></td><td colspan="1"><b>Counts</b></td></tr>
    <tr><td colspan="1"><b>GSE120575</b></td><td colspan="1">scRNA-seq</td><td colspan="1">2725</td><td colspan="1">3203</td><td colspan="1">5928</td><td colspan="1"><p>PD1, CTLA4+PD1,</p><p>CTLA4 (baseline); PD1 (post I and II)</p><p>CTLA4 (baseline); PD1 (post I)</p></td><td colspan="1">Sade-Feldman et al.</td></tr>
    <tr><td colspan="1"><b>Riaz</b> </td><td colspan="1">RNA-seq</td><td colspan="1">18</td><td colspan="1">31</td><td colspan="1">49</td><td colspan="1"><p>Anti-PD1 without previous anti-CTLA4,</p><p>Anti-PD1 with previous anti-CTLA4</p></td><td colspan="1">Riaz et al.</td></tr>
    <tr><td colspan="1"><b>Van</b> </td><td colspan="1">RNA-seq</td><td colspan="1">12</td><td colspan="1">29</td><td colspan="1">41</td><td colspan="1">anti-CTLA4 monotherapy</td><td colspan="1">Van et al.</td></tr>
    <tr><td colspan="1"><b>Hugo</b> </td><td colspan="1">RNA-seq</td><td colspan="1">15</td><td colspan="1">12</td><td colspan="1">27</td><td colspan="1">Anti-PD1 monotherapy </td><td colspan="1">Hugo et al.</td></tr>
    <tr><td colspan="1"><b>Lee</b></td><td colspan="1">RNA-seq</td><td colspan="1">22</td><td colspan="1">22</td><td colspan="1">44</td><td colspan="1">Anti-PD1 monotherapy</td><td colspan="1">Lee et al.</td></tr>
    <tr><td colspan="1"><b>Gide</b></td><td colspan="1">RNA-seq</td><td colspan="1">45</td><td colspan="1">27</td><td colspan="1">72</td><td colspan="1"><p>Anti-PD1 monotherapy,</p><p>Anti-PD1 + anti-CTLA4 treatment</p></td><td colspan="1">Gide et al.</td></tr>
    <tr><td colspan="1"><b>MGH</b></td><td colspan="1">RNA-seq</td><td colspan="1">6</td><td colspan="1">13</td><td colspan="1">19</td><td colspan="1"><p>Anti-PD1 monotherapy,</p><p>Anti-PDL1 monotherapy,</p><p>Anti-PD1+anti-CTLA4</p></td><td colspan="1">Auslander et al.</td></tr>
    <tr><td colspan="1"><b>Bulk Total</b></td><td colspan="1"></td><td colspan="1"><b>118</b></td><td colspan="1"><b>134</b></td><td colspan="1"><b>252</b></td><td colspan="1"></td><td colspan="1"></td></tr>
    </table>
- **NCBI_Melanoma_related_genes:** [NCBI_Melanoma_gene.txt](data/Pred/NCBI_Melanoma_gene.txt)
### code
- [ImmunotherapyResponsePrediction.ipynb](ImmunotherapyResponsePrediction.ipynb)

## skin cancer diagnosis

### datasets
- **cohorts:**  **Microarray** 
    - **RMA:** [data_rma.csv.gz](data/Diag/microarray/data_rma.csv.gz)
    - **RMA+debat:** [data_debat.csv.gz](data/Diag/microarray/data_debat.csv.gz)
    - **RMA+norm:** [data_norm.csv.gz](data/Diag/microarray/data_norm.csv.gz)
    - **anno:** [data_labels.csv](data/Diag/microarray/data_labels.csv)

    **Table 2: Datasets information in skin cancer classification**

    |**Datasets**|**Technology**|**Annotation data chip**|**Healthy**|**BCC**|**SCC**|**MCC**|**MEL**|**Count**|
    | - | - | - | - | - | - | - | - | - |
    |**GSE02503**|Affymetrix|hgu133a.db|6||5|||**11**|
    |**GSE03189**|Affymetrix|hgu133a.db|25||||45|**70**|
    |**GSE06710**|Affymetrix|hgu133a.db|13|||||**13**|
    |**GSE07553**|Affymetrix|hgu133plus2.db|4|15|11||56|**86**|
    |**GSE13355**|Affymetrix|hgu133plus2.db|64|||||**64**|
    |**GSE14905**|Affymetrix|hgu133plus2.db|21|||||**21**|
    |**GSE15605**|Affymetrix|hgu133plus2.db|16||||58|**74**|
    |**GSE29359**|Illumina|illuminaHumanv2.db|||||82|**82**|
    |**GSE30999**|Affymetrix|hgu133plus2.db|85|||||**85**|
    |**GSE32407**|Affymetrix|hgu133a2.db|10|||||**10**|
    |**GSE32628**|Illumina|lumiHumanAll.db|||15|||**15**|
    |**GSE32924**|Affymetrix|hgu133plus2.db|8|||||**8**|
    |**GSE36150**|Affymetrix|huex10sttranscriptcluster.db||||15||**15**|
    |**GSE39612**|Affymetrix|hgu133plus2.db||2|4|30||**36**|
    |**GSE42109**|Affymetrix|hgu133a2.db||11||||**11**|
    |**GSE42677**|Affymetrix|hgu133a2.db/hgu133plus2.db|10||10|||**20**|
    |**GSE45216**|Affymetrix|hgu133plus2.db|||30|||**30**|
    |**GSE46517**|Affymetrix|hgu133a.db|16||||83|**99**|
    |**GSE50451**|Affymetrix|hgu133plus2.db||||23||**23**|
    |**GSE52471**|Affymetrix|hgu133a2.db|13|||||**13**|
    |**GSE53223**|Affymetrix|hgu133plus2.db|18|||||**18**|
    |**GSE53462**|Illumina|lumiHumanAll.db|5|16|5|||**26**|
    |**GSE66359**|Affymetrix|hgu133plus2.db|||8|||**8**|
    |**GSE82105**|Affymetrix|hgu133plus2.db|6||||6|**12**|
    |**count**|||**320**|**44**|**88**|**68**|**330**|**850**|

    BCC = Basal cell carcinoma, SCC = Squamous cell carcinoma, MCC = Merkel cell carcinoma, MEL = melanoma
    
    **Table 3: Classifications for the datasets in skin cancer diagnosis**

    <table><tr><th colspan="1" valign="top"><b>2&nbsp;-&nbsp;class</b></th><th colspan="1" valign="top"><b>3&nbsp;-&nbsp;class</b></th><th colspan="1" valign="top"><b>5&nbsp;-&nbsp;class</b></th><th colspan="1" valign="top"><b>Discovery Sets (Training set)</b></th><th colspan="1" valign="top"><b>Validation Sets</b></th></tr>
    <tr><td colspan="3">Healthy</td><td colspan="1" valign="top">GSE30999 (85), GSE13355 (64), GSE03189 (25)</td><td colspan="1" valign="top">GSE02503 (6), GSE06710 (13), GSE07553 (4), GSE14905 (21), GSE15605 (16), GSE32407 (10), GSE32924 (8), GSE42677 (10), GSE52471 (13), GSE53462 (5), GSE82105 (6), GSE46517 (16), GSE53223 (18)</td></tr>
    <tr><td colspan="1" rowspan="4">Disease</td><td colspan="1" rowspan="3">Non&nbsp;-&nbsp;MEL</td><td colspan="1">BCC</td><td colspan="1" valign="top">GSE07553 (15), GSE42109 (11)</td><td colspan="1" valign="top">GSE39612 (2), GSE53462 (16)</td></tr>
    <tr><td colspan="1">SCC</td><td colspan="1" valign="top">GSE45216 (30), GSE07553 (11)</td><td colspan="1" valign="top"><p>GSE02503 (5), GSE32628 (15), GSE39612 (4)</p><p>GSE42677 (10), GSE53462 (5), GSE66359 (8)</p></td></tr>
    <tr><td colspan="1">MCC</td><td colspan="1" valign="top">GSE39612 (30)</td><td colspan="1" valign="top">GSE36150 (15), GSE50451 (23)</td></tr>
    <tr><td colspan="2">MEL</td><td colspan="1" valign="top">GSE07553 (56), GSE15605 (58), GSE03189 (45)</td><td colspan="1" valign="top">GSE29359 (82), GSE46517 (83), GSE82105 (6)</td></tr>
    <tr><td colspan="3"><b>Count</b></td><td colspan="1" valign="top"><b>430</b></td><td colspan="1" valign="top"><b>420</b></td></tr>
    </table>

- **NCBI_skin_cancer_gene:** [NCBI_skin_cancer_gene.txt](data/Diag/NCBI_skin_cancer_gene.txt)

### code
- [SkinCancerDiagnosis.ipynb](SkinCancerDiagnosis.ipynb)