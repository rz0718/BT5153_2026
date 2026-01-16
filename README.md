# Code Repo BT5153 2026 Spring


<br><br>


### Google Colab:

1. Follow this Notebook installation :
https://colab.research.google.com/drive/17SKoFRagHWZisth7iwWPrkNGsXQ78uGz?usp=sharing

2. Open your Google Drive :
https://www.google.com/drive

3. Open in Google Drive Folder 'BT5153_2026' and go to Folder 'BT5153_2026/codes/'
Select the notebook 'notebookname.ipynb' and open it with Google Colab using Control Click + Open With Colaboratory

4. If we need to load datasets from gdrive, we should run the following code at the beginning to specifiy the path

```python
### if we are using google colab, we need to run this cell to specify the path for data loading
import sys, os
if 'google.colab' in sys.modules:
    # mount google drive
    from google.colab import drive
    drive.mount('/content/gdrive')
    # specify the path of the folder containing "file_name" by changing the lecture index:
    lecture_index = '01'
    path_to_file = '/content/gdrive/My Drive/BT5153_2026/codes/lab_lecture{}/'.format(lecture_index)
    print(path_to_file)
    # change current path to the folder containing "file_name"
    os.chdir(path_to_file)
    !pwd
```
<br><br>