# AUTO ML APP: Developing and Deploying Dash via Heroku

Creating aN AUTO ML Dash from a pandas dataframe. Includes classification and time series alorithms. The app implements an explained-ML philosophy. Classification includes dimensionality reduction with t-sne for visualization and PCA. Classification uses simple Logistic Regression and Support Linear Vector. Time series include autocorrelation and XGBoost Regression. All algorithms intent to be explained. 

Web app: https://auto-ml-app.herokuapp.com/

To deploy an app like this: 

1. Setup account on Heroku
2. Create a new app and deploy using GitHub. Connect to your Github repo where app files are. 
3. As alternative, deploy app with [The Heroku CLI](https://devcenter.heroku.com/articles/heroku-cli#getting-started)
4. Install [Heroku CLI](https://devcenter.heroku.com/articles/heroku-cli#download-and-install)
5. Change to directory where your app files are:  
    ```
    $ cd ~/myapp
    ```
6. Login in Heroku:
    ```
    $ heroku login  
    ```
7. Create an app: 
    ```
    $ heroku create
    ```
8. Initialize a git repository in a new or existing directory with:
    ```
    $ git init 
    
    $ heroku git:remote -a <app-name>
    ```
9. Deploy application:
   ``` 
   $ git add .
   
   $ git commit -am "make it better"
   
   $ git push heroku master
    ```

