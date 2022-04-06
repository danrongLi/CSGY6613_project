#CS-GY6613 Project -- Team 20
##Team members:

* Danrong Li dl4111@nyu.edu - (AWS contact person)

* Yuxiong (Cody) Cheng yc4909@nyu.edu

* Xuanhua Zhang xz3426@nyu.edu

## Table of Contents

1. [OverView](##OverView)
2. [BaseLine](##BaseLine)
3. [Improvement](#third-example)
4. [Citation](###Citation)

## OverView
In many domains we are interested in finding artifacts that are similar to a query artifact. In this project you are going to implement a system that can find artifacts when the queries are visual. The type of queries you will test your system on are images of faces (30 points) and videos of people. You will also document your approach clearly showing advantages over the baseline.

In the following we use the term person of interest (PoI) to indicate the person that we reverse search on images and videos. The PoI may be present together with others in the datasets but the system should be able to retrieve the images / videos that the PoI. See bonus points for partial occlusion.


## BaseLine
### Quick Start
**Clone the project**
 ``` console
 git clone https://github.com/danrongLi/CSGY6613_project.git
 ```
 **Connect Google Colab to your local kernal**


``` console
pip install --upgrade jupyter_http_over_ws>=0.0.7 && jupyter serverextension enable --py jupyter_http_over_ws
jupyter notebook --NotebookApp.allow_origin='https://colab.research.google.com' --port=8889 --NotebookApp.port_retries=0
```

**Copy the URL and paste it in your google colab**
![image](pics/Screen%20Shot%202022-04-06%20at%207.10.39%20PM.png)


If you want to simulate running locally, please remember to copy and paste the folder "Cody" to your local google colab; delete all pickle (4) and h5 (1) files

Also change all file paths inside jupyter ==(replace r'XX\XX' with your own google drive file path)==



### Citation
- Practical Deep Learning for Cloud, Mobile, and Edge 
https://www.oreilly.com/library/view/practical-deep-learning/9781492034858/
