#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
#include <random>
#include <string>

#define _A 1
#define _B 1
#define _C 0.6
#define tau 1.0
#define dt 0.01
#define dim2vector vector<vector<double>>
#define dim3vector vector<vector<vector<double>>>
#define dim4vector vector<vector<vector<vector<double>>>>
using namespace std;

// 都市配置
double city[10][2] = {
	{0.125,0.9},
	{0.55,0.925},
	{0.9,0.85},
	{0.85,0.8},
	{0.8,0.15},
	{0.05,0.075},
	{0.125,0.175},
	{0.025,0.425},
	{0.35,0.55},
	{0.4,0.7}
};

int d(int i, int j, int n){
	if(j == -1){
		j = n - 1;
	}else if(j == n){
		j = 0;
	}
	if(i == j){
		return 1;
	}else{
		return 0;
	}
}


double dist(int i, int j){	
	return sqrt((city[i][0]-city[j][0])*(city[i][0]-city[j][0])+(city[i][1]-city[j][1])*(city[i][1]-city[j][1]));
}

// 最適解の総距離(上の都市配置の厳密解を天下り的に求める)
double minDistance(int N){
	double total_dist = 0;
	for(int i = 0; i < N; i++){
		total_dist += (i == N-1)? dist(i,0) : dist(i,i+1);
	}
	return total_dist;
}

void setWeight(dim4vector &w){
	int n =  w.size();
	for(int i = 0; i < n; i++){
		for(int j = 0; j < n; j++){
			for(int k = 0; k < n; k++){
				for(int l = 0; l < n; l++){
					w[i][j][k][l] = -_A*(d(i,k,n)+d(j,l,n))  +  _B*d(i,k,n)*d(j,l,n)  -  _C*dist(i,k)*(d(l,j+1,n)+d(l,j-1,n));
				}
			}
		}
	}
}

void initializeState(dim2vector &u){
	int n = u.size();
	random_device rd; /* 乱数生成器 */
	mt19937 mt(rd()); /* メルセンヌ・ツイスタ生成器 */
	uniform_real_distribution<double> rand(-0.02, 0.02); 
	for(int i = 0; i < n; i++){
		for(int j = 0; j < n; j++)
			u[i][j] = rand(mt);
	}
}

void sigmoid(dim2vector &x, dim2vector &u){
	double beta = 40.0;
	int n = x.size();

	for(int i = 0; i < n; i++){
		for(int j = 0; j < n; j++){
			x[i][j] = (tanh(beta*u[i][j])+1.0) / 2.0;
		}
	}
}

void renewState(dim4vector &w, dim2vector &u, dim2vector &x){
	int n = u.size();
	double h = 2*_A-(_B/2.0);
	for(int i = 0; i < n; i++){
		for(int j = 0; j < n; j++){
			u[i][j] = (1-(dt/tau)) * u[i][j] + (dt/tau) * h;
			for(int k = 0; k < n; k++){
				for(int l = 0; l < n; l++){
					u[i][j] += (dt/tau)*w[i][j][k][l]*x[k][l];
				}
			}
		}
	}
}

void viewState(dim2vector &array){
	int n =  array.size();
	for(int i = 0; i < n; i++){
		for(int j = 0; j < n; j++){
			cout << setw(10) << right << setprecision(2) << array[i][j] << ((j==n-1)?"":",");
		}
		cout << endl;
	}
}

void viewWeight(dim4vector &w){
	int n = w.size();
	for(int i = 0; i < n; i++){
		for(int j = 0; j < n; j++){
			for(int k = 0; k < n; k++){
				for(int l = 0; l < n; l++){
					cout << "w["<<i<<"]["<<j<<"]["<<k<<"]["<<l<<"] = "<< w[i][j][k][l]<<endl;
				}
			}
		}
	}
}

void shapeTempArray(int N, dim2vector &prevu, dim2vector &prevx){
	prevu.resize(N);
	prevx.resize(N);
	for(int i = 0; i < N; i++){
		prevu[i].resize(N);
		prevx[i].resize(N);
	}
	for(int i = 0; i < N; i++){
		for(int j = 0; j < N; j++){
			prevu[i][j] = 0;
			prevx[i][j] = 0;
		}
	}
}

bool isConvergence(int *cnt, dim2vector &u, dim2vector &prevu, dim2vector &x, dim2vector &prevx){
	int n = u.size();
	bool flag = true;

	for(int i = 0; i < n; i++){
		for(int j = 0; j < n; j++){
			if( abs(x[i][j] - prevx[i][j]) > 1e-12 && abs(u[i][j] - prevu[i][j]) > 1e-12){
				flag = false;
			}
		}
	}
	if(flag == true){
		(*cnt)++;
	}
	return *cnt >= 10;
}

void copyArray(dim2vector &array1, dim2vector &array2){
	int n = array1.size();
	
	for(int i = 0; i < n; i++){
		for(int j = 0; j < n; j++){
			array2[i][j] = array1[i][j];
		}
	}
}

void neural(dim4vector &w, dim2vector &u, dim2vector &x){
	int N = x.size();

	// 収束判定用変数
	int cnt = 0;
	bool convFlag = false;
	dim2vector prevu, prevx;
	
	// 内部状態u,出力xの初期値設定
	initializeState(u);
	sigmoid(x,u);
	shapeTempArray(N, prevu, prevx);

	// 内部状態と出力の更新
	for(double t = 0.0; t < 200*tau; t += dt){
		renewState(w,u,x);
		sigmoid(x,u);
		
		// 収束判定
		convFlag = isConvergence(&cnt, u, prevu, x, prevx);
		copyArray(u, prevu);
		copyArray(x, prevx);
		if(convFlag){
			break;
		}
		
	}
}

bool isValidTour(dim2vector x){
	int n = x.size();
	int count;
	double temp;

	// 0.5以上を1とする
	for(int i = 0; i < n; i++){
		for(int j = 0; j < n; j++){
				x[i][j] = (x[i][j] > 0.5) ? 1 : 0; 
		}
	}	
	// 横に1があればfalse
	for(int i = 0; i < n; i++){
		count = 0;
		for(int j = 0; j < n; j++){
			if(x[i][j] == 1){
				count++;
			}
		}
		if(count != 1){
			return false;
		}
	}
	// 縦に1があればfalse
	for(int i = 0; i < n; i++){
		count = 0;
		for(int j = 0; j < n; j++){
			if(x[j][i] == 1){
				count++;
			}
		}
		if(count != 1){
			return false;
		}
	}
	return true;
}

double calcTotalDistance(dim2vector x){
	int n =  x.size();
	double distance = 0;

	// 0.5以上を1とする
	for(int i = 0; i < n; i++){
		for(int j = 0; j < n; j++){
				x[i][j] = (x[i][j] > 0.5) ? 1 : 0; 
		}
	}
		
	for(int i = 0; i < n; i++){
		for(int j = 0; j < n; j++){
			for(int k = 0; k < n; k++){
				for(int l = 0; l < n; l++){
					distance += dist(i,k)*d(l,j+1,n)*x[i][j]*x[k][l];
				}
			}
		}
	}
	return distance;
}

void shapeArray(int N, dim4vector &w, dim2vector &u, dim2vector &x){
	u.resize(N);
	x.resize(N);
	w.resize(N);
	for(int i = 0; i < N; i++){
		u[i].resize(N);
		x[i].resize(N);
		w[i].resize(N);
		for(int j = 0; j < N; j++){
			w[i][j].resize(N);
			for(int k = 0; k < N; k++){
				w[i][j][k].resize(N);
			}
		}
	}
}

int main(int argc,char *argv[])
{
	const int N = 10; // 問題サイズ
	double total_dist = 0;
	double min_dist = 0;
	dim2vector u;
	dim2vector x;
	dim4vector w;

	shapeArray(N,w,u,x); // 配列の形を決める
	setWeight(w); // 問題設定(荷重値の設定)
	
	neural(w,u,x);
	
	viewState(x);
	total_dist = calcTotalDistance(x);
	min_dist = minDistance(N);
	cout << "Minimal distance(in the optimum solution): " << min_dist << endl;
	cout << "Total distance(in the approximate solution): " << total_dist << endl;
	cout << "Valid tour?: " << ((isValidTour(x)) ? "True" : "False") << endl;
	cout << "Suboptimal solution?: " << ((isValidTour(x) && (min_dist * 1.2 >= total_dist)) ? "True" : "False") << endl;
	
	return 0;
}