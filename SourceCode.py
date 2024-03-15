import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#-------------load du lieu va chia du lieu-----------
from sklearn.model_selection import train_test_split
# ----------------------mo hinh -----------
import sklearn
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor
#--------Danh gia mo hinh 
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error as MAE
from sklearn.metrics import r2_score
#--------Thoi gian chay chuong trinh
import time
start_time = time.time()
#============================================

#------------------------
#Đọc dữ liệu từ File
dulieuLoad = pd.read_csv("wankara.csv")

dulieu_X = dulieuLoad.iloc[:,0:-1]
print(dulieu_X)
dulieu_Y =  dulieuLoad.iloc[:,-1]#Doc cot cuoi cung
print("dulieu Y:",dulieu_Y)

#----------------------
#Tham số:
print("Số lượng phần tử",len(dulieu_X))
#----------------------
# Ham de ve bieu do
def estimate_coef(x, y): 
    # number of observations/points 
    n = np.size(x) 
  
    # mean of x and y vector 
    m_x, m_y = np.mean(x), np.mean(y) 
  
    # calculating cross-deviation and deviation about x 
    SS_xy = np.sum(y*x) - n*m_y*m_x 
    SS_xx = np.sum(x*x) - n*m_x*m_x 
  
    # calculating regression coefficients 
    b_1 = SS_xy / SS_xx 
    b_0 = m_y - b_1*m_x 
  
    return(b_0, b_1) 
  
def plot_regression_line(x, y, b): 
    # plotting the actual points as scatter plot 
    plt.scatter(x, y, color = "m", 
               marker = "o", s = 30) 
  
    # predicted response vector 
    y_pred = b[0] + b[1]*x 
  
    # plotting the regression line 
    plt.plot(x, y_pred, color = "g") 
  
    # putting labels 
    
    plt.xlabel('Gia tri thuc te') 
    plt.ylabel('Gia tri du doan') 
  
    # function to show plot 
    plt.show() 
#---------------------
# Hàm để tính giá trị trung bình 
def cal_average(num):
    sum_num = 0
    for t in num:
        sum_num = sum_num + t           

    avg = sum_num / len(num)
    return avg
#==========================================
#Chia tập dữ liệu

X_Train, X_Test, Y_Train, Y_Test = train_test_split(dulieu_X,dulieu_Y, test_size=1/3.0, random_state=10)

print(Y_Test)
print("So luong Train:", len(X_Train))
print("So luong Test:", len(X_Test))

#========================================================
n = 0
ypred_50lan_DecisionTreeReg = []
ypred_50lan_LinearRegression = []
ypred_50lan_Lasso = []
ypred_50lan_RandomForestRegressor = []
while (n < 50):
    n += 1
    #Load mo hinh DecisionTreeRegressor
    from sklearn.ensemble import BaggingRegressor
    from sklearn.tree import DecisionTreeRegressor
    tree = DecisionTreeRegressor(random_state= 0)

    bagging_regtree = BaggingRegressor(estimator=tree, n_estimators=10,random_state=42)
    bagging_regtree.fit(X_Train,Y_Train)

        #Du doan mo hinh DecisionTreeRegressor
    Y_DuDoan = bagging_regtree.predict(X_Test)
        # print("Ket qua du doan DecisionTreeRegressor: ",Y_DuDoan)
    ypred_50lan_DecisionTreeReg.append(Y_DuDoan)
    #------------------------------------------------------------------------
    #Load mo hinh LinearRegression
    lm = linear_model.LinearRegression()
    lm.fit(dulieu_X,dulieu_Y)
    #Du doan mo hinh LinearRegression
    y_pred = lm.predict(X_Test)
    ypred_50lan_LinearRegression.append(y_pred)

    
    #------------------------------------------------------------------------
    #Load mo hinh Lasso
    Lasso_reg = linear_model.Lasso(alpha=0.5)
    Lasso_reg.fit(dulieu_X,dulieu_Y)
    #Du doan mo hinh Lasso
    y_pred_Lasso = Lasso_reg.predict(X_Test)
    ypred_50lan_Lasso.append(y_pred)

    #------------------------------------------------------------------------


    #Load mo hinh RandomForestRegressor
    RandomForest = RandomForestRegressor(n_estimators=100, random_state=0)
    RandomForest.fit(dulieu_X,dulieu_Y)
    #Du doan mo hinh RandomForestRegressor
    y_pred_RandomForest = RandomForest.predict(X_Test)
    ypred_50lan_RandomForestRegressor.append(y_pred)

    #------------------------------------------------------------------------
ypred_DecisionTreeReg = cal_average(ypred_50lan_DecisionTreeReg)
ypred_LinearRegression = cal_average(ypred_50lan_LinearRegression)
ypred_Lasso = cal_average(ypred_50lan_Lasso)
ypred_RandomForestRegressor = cal_average(ypred_50lan_RandomForestRegressor)

#========================================================
# vẽ biễu đồ so sánh kết quả y dự đoán và y thực tế
def main(): 
    plt.title('mo hinh DecisionTreeRegressor')
    # observations 
    x = Y_Test
    y = ypred_DecisionTreeReg 
  
    # estimating coefficients 
    b = estimate_coef(x, y) 
    
  
    # plotting regression line 
    plot_regression_line(x, y, b)

if __name__ == "__main__": 
    main()


def main(): 
    plt.title('mo hinh LinearRegression')
    # observations 
    x = Y_Test
    y = ypred_LinearRegression 
  
    # estimating coefficients 
    b = estimate_coef(x, y) 
    
  
    # plotting regression line 
    plot_regression_line(x, y, b) 

if __name__ == "__main__": 
    main() 

def main(): 
    plt.title('mo hinh Lasso')
    # observations 
    x = Y_Test
    y = ypred_Lasso 
  
    # estimating coefficients 
    b = estimate_coef(x, y) 
    
  
    # plotting regression line 
    plot_regression_line(x, y, b) 

if __name__ == "__main__": 
    main() 

def main(): 
    plt.title('mo hinh RandomForestRegressor')
    # observations 
    x = Y_Test
    y = ypred_RandomForestRegressor 
  
    # estimating coefficients 
    b = estimate_coef(x, y) 
    
  
    # plotting regression line 
    plot_regression_line(x, y, b) 

if __name__ == "__main__": 
    main() 
#=========================================================
#Danh Gia THông qua chỉ so MSE, RMSE, MEAN - DecisionTreeRegressor
err_DecisionTreeRegressor = mean_squared_error(Y_Test, Y_DuDoan)
err_Decision = mean_squared_error(Y_Test, ypred_DecisionTreeReg)
print("Danh Gia Thông qua chỉ so MSE và RMSE DecisionTreeRegressor")
print("MSE",round(err_DecisionTreeRegressor,3))
print("RMSE %.3f" %(np.sqrt(err_DecisionTreeRegressor)))
MEAN = MAE(Y_Test, Y_DuDoan)
print("MEAN",MEAN)
print("R2 score : %.2f" % r2_score(Y_Test, Y_DuDoan))

#Danh Gia THông qua chỉ so MSE, RMSE, MEAN - LinearRegression
err_LinearRegression = mean_squared_error(Y_Test, y_pred)
print("Danh Gia Thông qua chỉ so MSE và RMSE LinearRegression")
print("MSE",round(err_LinearRegression,3))
print("RMSE %.3f" %(np.sqrt(err_LinearRegression)))
MEAN = MAE(Y_Test, y_pred)
print("MEAN",MEAN)
print("R2 score : %.2f" % r2_score(Y_Test, y_pred))


#Danh Gia THông qua chỉ so MSE, RMSE, MEAN - Lasso
err_Lasso = mean_squared_error(Y_Test, y_pred_Lasso)
print("Danh Gia Thông qua chỉ so MSE và RMSE Lasso")
print("MSE",round(err_Lasso,3))
print("RMSE %.3f" %(np.sqrt(err_Lasso)))
MEAN = MAE(Y_Test, y_pred_Lasso)
print("MEAN",MEAN)
print("R2 score : %.2f" % r2_score(Y_Test, y_pred_Lasso))

#Danh Gia THông qua chỉ so MSE, RMSE, MEAN - RandomForest
err_RandomForest = mean_squared_error(Y_Test, y_pred_RandomForest)
print("Danh Gia Thông qua chỉ so MSE và RMSE RandomForest")
print("MSE",round(err_RandomForest,3))
print("RMSE %.3f" %(np.sqrt(err_RandomForest)))
MEAN = MAE(Y_Test, y_pred_RandomForest)
print("MEAN",MEAN)
print("R2 score : %.2f" % r2_score(Y_Test, y_pred_RandomForest))
#========================================================
#Thoi gian chuong trinh chay 
print("========================================================")
end_time = time.time()
print("Thời gian máy tính 2 thực hiện: {:.3f} seconds".format(end_time - start_time))


