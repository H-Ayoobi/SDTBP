//============================================================================
// Name        : BP_custom.cpp
// Author      : csvans
// Version     :
// Copyright   : hamed_ayoobi
// Description : Hello World in C++, Ansi-style
//============================================================================

#include <iostream>
#include <cv.hpp>
#include <highgui.hpp>
#include <imgproc.hpp>
#include <iostream>
#include <math.h>
#include <fstream>
#include <sstream>
#include <time.h>

using namespace std;
using namespace cv;

int min(int size, int* in) {

	int minVal = 10000000;
	for (int i = 0; i < size; ++i) {
		if (in[i] < minVal)
			minVal = in[i];
	}
	return minVal;
}

Mat makeModelAndInference(Mat image, Mat filterIm, int iter) {
	Mat OrIm = image; //original image
	Mat EIm = image.clone(); //edited image
	int width = image.cols;
	int height = image.rows;
	bool haveDenoising = true;
	//message= min( smoothness cost + data cost + sum over other incoming messages)

	vector<int*> Lmessages((height * width) - height); // left outgoing message of node
	vector<int*> Rmessages((height * width) - height); // right outgoing message of node
	vector<int*> Dmessages((height * width) - width); // up outgoing message of node
	vector<int*> Umessages((height * width) - width); // down outgoing message of node

	for (int i = 0; i < ((height * width) - height); ++i) {
		Lmessages[i] = new int[256];
		Rmessages[i] = new int[256];
		for (int k = 0; k < 256; ++k) {
			Lmessages[i][k] = 0;
			Rmessages[i][k] = 0;
		}
	}
	for (int i = 0; i < ((height * width) - width); ++i) {

		Umessages[i] = new int[256];
		Dmessages[i] = new int[256];
		for (int k = 0; k < 256; ++k) {
			Umessages[i][k] = 0;
			Dmessages[i][k] = 0;
		}
	}
	int costD, costU, costR, costL;
	int Dcost;
	clock_t TwoIterationTotalTime;
	clock_t totalStartTime = clock();
	clock_t temp=0;
	int lasting=0;

	int minccount=10000000;
	int minEnergy=1000000000;
	int s=1;
	for (int t = 0; t<iter; ++t) {
		clock_t _startTime = clock();

		if(t%2==0)
			TwoIterationTotalTime=clock();
//		cout << "iteration number " << t + 1 << " started." << endl; // prints !!!Hello World!!!
		for (int i = 0; i < height; i++)
			for (int j = 0; j < width; ++j) {
				int totalMinDScostD = 1000000;
				int totalMinDScostU = 1000000;
				int totalMinDScostL = 1000000;
				int totalMinDScostR = 1000000;
				int minMD=100000;
//				int mMD=100000;
//				int minMU=100000;
//				int mMU=100000;
//				int minML=100000;
//				int mML=100000;
//				int minMR=100000;
//				int mMR=100000;
				if ((i + j) % 2 == t % 2) {
					//********** message computation with distance transform ************
					int mD[256] = { 0 };
					int mU[256] = { 0 };
					int mR[256] = { 0 };
					int mL[256] = { 0 };
					int tempCostD=-1;
					for (int k2 = 0; k2 < 256; ++k2) //these are the values for labels of p
							{
						if ((int) (filterIm.at<uchar>(i, j)) < 230)
							Dcost = 0;
						else
							//data cost
							Dcost = abs(k2 - (int) (OrIm.at<uchar>(i, j)));

						costD = Dcost;
						costU = Dcost;
						costR = Dcost;
						costL = Dcost;

						//other messages
						if (j > 0) {
							costD += Rmessages[(height * (j - 1)) + i][k2]; //the incoming right message
							costU += Rmessages[(height * (j - 1)) + i][k2]; //the incoming right message
							costR += Rmessages[(height * (j - 1)) + i][k2]; //the incoming right message
						}
						if (i > 0) {
							costD += Dmessages[(width * (i - 1)) + j][k2]; //the incoming down message
							costR += Dmessages[(width * (i - 1)) + j][k2]; //the incoming down message
							costL += Dmessages[(width * (i - 1)) + j][k2]; //the incoming down message
						}
						if (j < width - 1) {
							costD += Lmessages[(height * (j)) + i][k2]; //the incoming left message
							costU += Lmessages[(height * (j)) + i][k2]; //the incoming left message
							costL += Lmessages[(height * (j)) + i][k2]; //the incoming left message
						}
						if (i < height - 1) {
							costU += Umessages[(width * (i)) + j][k2]; //the incoming up message
							costR += Umessages[(width * (i)) + j][k2]; //the incoming up message
							costL += Umessages[(width * (i)) + j][k2]; //the incoming up message
						}
						mD[k2] = costD;
						mU[k2] = costU;
						mR[k2] = costR;
						mL[k2] = costL;
//						if(costD+k2<minMD)
//							minMD=costD+k2;
//						if(costD-k2<mMD )
//							mMD=costD-k2;
//						if(costU+k2<minMU)
//							minMU=costU+k2;
//						if(costU-k2<mMU)
//							mMU=costU-k2;
//						if(costR+k2<minMR)
//							minMR=costR+k2;
//						if(costR-k2<mMR)
//							mMR=costR-k2;
//						if(costL+k2<minML)
//							minML=costL+k2;
//						if(costL-k2<mML)
//							mML=costL-k2;
//						tempCostD=costD;

					} //end for k2
					  //till this point m array initialized with h values

//					int tempDmin=10000;
//					int tempUmin=10000;
//					int tempRmin=10000;
//					int tempLmin=10000;
//					int d=150;
//					for(int k=0;k<256;++k)
//					{
//						if(mD[k]+d<tempDmin)
//							tempDmin=mD[k]+d;
//						if(mU[k]+d<tempUmin)
//							tempUmin=mU[k]+d;
//						if(mR[k]+d<tempRmin)
//							tempRmin=mR[k]+d;
//						if(mL[k]+d<tempLmin)
//							tempLmin=mL[k]+d;
//					}


					  //forward pass of distance transform formula

					for (int k = 1; k < 256; ++k) {
						mD[k] = min(mD[k], mD[k - 1] + s);//m[fq]=min(m[fq],m[fq-1]+s);
						mU[k] = min(mU[k], mU[k - 1] + s); //m[fq]=min(m[fq],m[fq-1]+s);
						mL[k] = min(mL[k], mL[k - 1] + s); //m[fq]=min(m[fq],m[fq-1]+s);
						mR[k] = min(mR[k], mR[k - 1] + s); //m[fq]=min(m[fq],m[fq-1]+s);
					}
					//backward pass in distance transform
					for (int k = 256 - 2; k >= 0; --k) {
						mD[k] = min(mD[k], mD[k + 1] + s); //m[fq]=min(m[fq],m[fq-1]+s);
						mU[k] = min(mU[k], mU[k + 1] + s); //m[fq]=min(m[fq],m[fq-1]+s);
						mL[k] = min(mL[k], mL[k + 1] + s); //m[fq]=min(m[fq],m[fq-1]+s);
						mR[k] = min(mR[k], mR[k + 1] + s); //m[fq]=min(m[fq],m[fq-1]+s);
					}
//					for(int k=0;k<256;++k)
//					{
//						mD[k]=min(mD[k],tempDmin);
//						mU[k]=min(mU[k],tempUmin);
//						mR[k]=min(mR[k],tempRmin);
//						mL[k]=min(mL[k],tempLmin);
//					}
//					int a,b,c,d;
//					bool too=false;
//					for (int k = 0; k < 256; ++k) {
//						a=max(-k+minMD,k+mMD);
//						b=max(-k+minMU,k+mMU);
//						c=max(-k+minMR,k+mMR);
//						d=max(-k+minML,k+mML);
//
//
//						if(a!=mD[k])
//							cout<<"iter:"<<t<<" i:"<<i<<" j:"<<j<<" k:"<<k<<" mD[k]:"<<mD[k]<<" a:"<<a<<endl;
//
//					}

			//		 now put complete message to messages vector
					for (int k = 0; k < 256; ++k) {

						if (i < height - 1)	//have down outgoing message
								{

							if (mD[k] < totalMinDScostD)
								totalMinDScostD = mD[k];
						}
						if (i > 0) //have up outgoing message
								{

							if (mU[k] < totalMinDScostU)
								totalMinDScostU = mU[k];
						}
						if (j > 0) //have left outgoing message
								{

							if (mL[k] < totalMinDScostL)
								totalMinDScostL = mL[k];
						}
						if (j < width - 1) //have right outgoing message
								{
							if (mR[k] < totalMinDScostR)
								totalMinDScostR = mR[k];
						}

					}
					//***************normalizing message*********************
					//normalizing message vector by minus the minimum value in each message vector from all the values
					for (int x = 0; x < 256; ++x) {

						if (i < height - 1) {
							Dmessages[(i * width) + j][x] = mD[x]
									- totalMinDScostD;

						}
						if (i > 0) {
							Umessages[((i - 1) * width) + j][x] = mU[x]
									- totalMinDScostU;
						}
						if (j > 0) {
							Lmessages[((j - 1) * height) + i][x] = mL[x]
									- totalMinDScostL;
						}
						if (j < width - 1) {
							Rmessages[(j * height) + i][x] = mR[x]
									- totalMinDScostR;

						}
					}//*****************end normalization of message**************
					 //************* end message computation ********************
				} //end if i+j%2==t%2
			} //end for i and for j
//		cout << "iteration-" << t + 1 << " execution time = "
//				<< (double) ((clock() - _startTime) / CLOCKS_PER_SEC)
//				<< " number of clocks() = " << (clock() - _startTime) << endl;




bool same=false;
int ccount=0;


		if(t%2==1)
		{
			//computing beliefs after message passing

//				for (int i = 0; i < height; ++i)
//					for (int j = 0; j < width; ++j) {
//						if((!haveDenoising && filterIm.at<uchar>(i, j) < 230) || haveDenoising)
//						{
//							int belief[256];
//							int minBelief = 1000000;
//							int minBeliefIndex = 0;
//							for (int k = 0; k < 256; ++k) {
//									if (filterIm.at<uchar>(i, j) > 230)
//										belief[k] = abs(k - (int) (OrIm.at<uchar>(i, j)));
//									else
//										belief[k] = 0;
//
//									if (!(i == 0)) //have up neighbor
//										belief[k] += Dmessages[(width * (i - 1)) + j][k];
//									if (!(i == height - 1)) //have down neighbor
//										belief[k] += Umessages[(width * (i)) + j][k];
//									if (!(j == 0)) //have left neighbor
//										belief[k] += Rmessages[(height * (j - 1)) + i][k];
//									if (!(j == width - 1)) //have right neighbor
//										belief[k] += Lmessages[(height * (j)) + i][k];
//
//								if (belief[k] < minBelief) {
//									minBelief = belief[k];
//									minBeliefIndex = k;
//								}
//
//							}
//
//					//		EIm.at<uchar>(i, j) = minBeliefIndex;
//
//						}
//					}

				//computing energy level of solution
					int energy = 0;
					int totalDcost = 0;
					int totalSMcost = 0;
					bool neighbors[4]; //0=up 1=right 2=down 3=left
					for (int i = 0; i < height; ++i)
						for (int j = 0; j < width; ++j) {
							if (filterIm.at<uchar>(i, j) > 230)
								totalDcost += abs(
										(int) (OrIm.at<uchar>(i, j))
												- (int) (EIm.at<uchar>(i, j)));
							neighbors[0] = false;
							neighbors[1] = false;
							neighbors[2] = false;
							neighbors[3] = false;
							if (i == 0) //don't have up neighbor
								neighbors[0] = true;
							else if (i == height - 1) //don't have down neighbor
								neighbors[2] = true;
							if (j == 0) //don't have left neighbor
								neighbors[3] = true;
							else if (j == width - 1) //don't have right neighbor
								neighbors[1] = true;

				//			if (neighbors[0] == false)
				//				totalSMcost += abs(
				//						(int) EIm.at<uchar>(i, j)
				//								- (int) EIm.at<uchar>(i - 1, j));
							if (neighbors[1] == false)
								totalSMcost += s * abs(
										(int) EIm.at<uchar>(i, j)
												- (int) EIm.at<uchar>(i, j + 1));
							if (neighbors[2] == false)
								totalSMcost += s * abs(
										(int) EIm.at<uchar>(i, j)
												- (int) EIm.at<uchar>(i + 1, j));
				//			if (neighbors[3] == false)
				//				totalSMcost += abs(
				//						(int) EIm.at<uchar>(i, j)
				//								- (int) EIm.at<uchar>(i, j - 1));

						}

					energy = totalDcost + totalSMcost;
					cout  << energy << endl;

//					if(energy<minEnergy)
//						minEnergy=energy;
//					else break;


					same=true;
					for (int i = 0; i < height; ++i)
							for (int j = 0; j < width; ++j) {
								int belief[256];
								int minBelief = 1000000;
								int minBeliefIndex = 0;
								for (int k = 0; k < 256; ++k) {

										if (filterIm.at<uchar>(i, j) > 230)
											belief[k] = abs(k - (int) (OrIm.at<uchar>(i, j)));
										else
											belief[k] = 0;

										if (!(i == 0)) //have up neighbor
											belief[k] += Dmessages[(width * (i - 1)) + j][k];
										if (!(i == height - 1)) //have down neighbor
											belief[k] += Umessages[(width * (i)) + j][k];
										if (!(j == 0)) //have left neighbor
											belief[k] += Rmessages[(height * (j - 1)) + i][k];
										if (!(j == width - 1)) //have right neighbor
											belief[k] += Lmessages[(height * (j)) + i][k];

										if (belief[k] < minBelief) {
											minBelief = belief[k];
											minBeliefIndex = k;
										}

								}
								int temp=EIm.at<uchar>(i,j);
								if(EIm.at<uchar>(i,j)!=minBeliefIndex){
									EIm.at<uchar>(i, j) = minBeliefIndex;
									same=false;
									ccount++;
								}
							}

					if(ccount<minccount)
					{
						minccount=ccount;
						lasting=0;
					}
					lasting++;
//					if(lasting>30)
//						same=true;

					cout<<ccount<<" - "<<lasting<<endl;


					temp+=(clock() - TwoIterationTotalTime);
							cout << temp << endl;

		}


		if(same==true)
		{
			cout<<"**converged with "<<t<<" iterations.";
			break;
		}
	} //end iteration loop
	cout << "all messeges sent after " << iter
			<< " iterations----now we compute beliefs and then edited image"
			<< endl;

	//computing beliefs after message passing

	for (int i = 0; i < height; ++i)
		for (int j = 0; j < width; ++j) {
			if((!haveDenoising && filterIm.at<uchar>(i, j) < 230) || haveDenoising)
			{
			int belief[256];
			int minBelief = 1000000;
			int minBeliefIndex = 0;
			for (int k = 0; k < 256; ++k) {

					if (filterIm.at<uchar>(i, j) > 230)
						belief[k] = abs(k - (int) (OrIm.at<uchar>(i, j)));
					else
						belief[k] = 0;

					if (!(i == 0)) //have up neighbor
						belief[k] += Dmessages[(width * (i - 1)) + j][k];
					if (!(i == height - 1)) //have down neighbor
						belief[k] += Umessages[(width * (i)) + j][k];
					if (!(j == 0)) //have left neighbor
						belief[k] += Rmessages[(height * (j - 1)) + i][k];
					if (!(j == width - 1)) //have right neighbor
						belief[k] += Lmessages[(height * (j)) + i][k];

				if (belief[k] < minBelief) {
					minBelief = belief[k];
					minBeliefIndex = k;
				}
			}
			EIm.at<uchar>(i, j) = minBeliefIndex;
			}
		}
	cout << "total execution time = "
			<< (double) ((clock() - totalStartTime) / CLOCKS_PER_SEC)
			<< " -- total number of clocks() = " << (clock() - totalStartTime)
			<< endl;

	//computing energy level of solution
	int energy = 0;
		int totalDcost = 0;
		int totalSMcost = 0;
		bool neighbors[4]; //0=up 1=right 2=down 3=left
		for (int i = 0; i < height; ++i)
			for (int j = 0; j < width; ++j) {
				if (filterIm.at<uchar>(i, j) > 230)
					totalDcost += abs(
							(int) (OrIm.at<uchar>(i, j))
									- (int) (EIm.at<uchar>(i, j)));
				neighbors[0] = false;
				neighbors[1] = false;
				neighbors[2] = false;
				neighbors[3] = false;
				if (i == 0) //don't have up neighbor
					neighbors[0] = true;
				else if (i == height - 1) //don't have down neighbor
					neighbors[2] = true;
				if (j == 0) //don't have left neighbor
					neighbors[3] = true;
				else if (j == width - 1) //don't have right neighbor
					neighbors[1] = true;

	//			if (neighbors[0] == false)
	//				totalSMcost += abs(
	//						(int) EIm.at<uchar>(i, j)
	//								- (int) EIm.at<uchar>(i - 1, j));
				if (neighbors[1] == false)
					totalSMcost += s * abs(
							(int) EIm.at<uchar>(i, j)
									- (int) EIm.at<uchar>(i, j + 1));
				if (neighbors[2] == false)
					totalSMcost += s * abs(
							(int) EIm.at<uchar>(i, j)
									- (int) EIm.at<uchar>(i + 1, j));
	//			if (neighbors[3] == false)
	//				totalSMcost += abs(
	//						(int) EIm.at<uchar>(i, j)
	//								- (int) EIm.at<uchar>(i, j - 1));
			}
		energy = totalDcost + totalSMcost;
		cout << "total Energy: " << energy << endl;

	return EIm;

}

int main(int argc, char** argv) {
	if (argc != 4) {
		std::cerr << "Usage: " << argv[0]
				<< " infilePath inpaintFilePath outFilePath" << std::endl;
		exit(-1);
	}

	char *infilename = argv[1];
	char *inpaintname = argv[2];
	char *outfilename = argv[3];
	Mat im;
	im = imread(infilename);
//	resize(im,im,Size(im.cols/2,im.rows/2));
	cvtColor(im, im, CV_BGR2GRAY);
	Mat origIm = im;
	Mat im2;
	im2 = imread(inpaintname);
//	resize(im2,im2,Size(im2.cols/2,im2.rows/2));
	cvtColor(im2, im2, CV_BGR2GRAY);
	Mat output = makeModelAndInference(im, im2, 800); //arguments: image,number of iterations

	imwrite(outfilename, output);
	cout << "!!!Hello World!!!" << endl; // prints !!!Hello World!!!
	return 0;

}
