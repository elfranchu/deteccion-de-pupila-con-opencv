// TestOpenCV.cpp: define el punto de entrada de la aplicación de consola.
//

#include "stdafx.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <string.h>

// vmd.cpp: define el punto de entrada de la aplicación de consola.
//
//#include "cv.h"
//#include "highgui.h" 
//#include "ml.h" 
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include "Auxiliar.h"
using namespace std;
	int numCARACTERISTICA=120;
	//90 peor resultado
 
 fstream ff;

struct numProb
{
	int numero;
	float prob;
};
void Muestra(vector<int>totalAciertosTest,int tam,bool entreno );
/** Booleana para ordenar probabilidades */
bool Compara(struct numProb np1,struct numProb np2)
{
	float f1=np1.prob;
	float f2=np2.prob;
	float result=f1-f2;
	return result > 0;
	return f1 >= f2;
}

int DameCoincidencias(vector<int>prediccion,vector<int>resultados)
{
	int aciertos=0;
	for(int k=0;k<prediccion.size();k++)
	{
		int numeropredicho=prediccion[k];
		for(int k2=0;k2<resultados.size();k2++)
			{ 
				if(resultados[k2]==numeropredicho)
					aciertos++;
			}
	}
	return aciertos;
}

/**actualiza el vector numProb */
void ActualizaSumas(int tamPrediccion,const vector<struct numProb>&numProbVect,
					vector<int>resultados,vector<int>&totalAciertosTest)
{
		vector<int>prediccion(0);
		for(int k=0;k<tamPrediccion;k++)
			prediccion.push_back(numProbVect[k].numero);
		int aciertos= DameCoincidencias( prediccion, resultados);
		 
			totalAciertosTest[aciertos]++;
 
}
//
//build_mlp_classifier2("rnfich.txt",0,
//					 "redneural.data",mlp, numCARACTERISTICA,DifChars,data,responses );
int build_mlp_classifier2( char* data_filename,
    char* filename_to_save, char* filename_to_load,CvANN_MLP mlp,int numCARACTERISTICAS,int DifChars ,CvMat* data,
    
    CvMat* responses  )
{
	fstream fdebug;
	fdebug.open("fdebug.txt",ios::out);
	int numcrahses=0;
	int numTOTAL=0;
	ff.open("LOGGGGG.txt",ios::out);
 
    const int class_count = DifChars;
    CvMat train_data;
    CvMat* mlp_response = 0;
	 CvMat* mlp_response2 = 0;
   
    int nsamples_all = 0, ntrain_samples = 0;
    int i, j;
    double train_hr = 0, test_hr = 0;
    //CvANN_MLP mlp;
	time_t start;
	time (&start);
	 
    printf( "The database %s is loaded.\n", data_filename );
    nsamples_all = data->rows;
    ntrain_samples = (int)(nsamples_all*0.8);
	fdebug<<"NTRAINSamples: "<<ntrain_samples<<"\n";

	 CvANN_MLP mlp2;
    // Create or load MLP classifier
    if(  filename_to_load )
    {
        // load classifier from the specified file
        mlp.load( filename_to_load );
          mlp2.load( filename_to_load );
        if( !mlp.get_layer_count() )
        {
            printf( "Could not read the classifier %s\n", filename_to_load );
            return -1;
        }
        printf( "The classifier %s is loaded.\n", data_filename );
    }
    else
    {
       CvMat* new_responses = cvCreateMat( ntrain_samples, class_count, CV_32F );
       printf( "Unrolling the responses...\n");
       for( i = 0; i < ntrain_samples; i++ )
        {
			float *ptroRespuesta=responses->data.fl+i*6;
			int primer_numero=cvRound(ptroRespuesta[0])-1 ;
			int segundo_numero=cvRound(ptroRespuesta[1]) -1 ;
			int tercer_numero=cvRound(ptroRespuesta[2])-1  ;
			int cuarto_numero=cvRound(ptroRespuesta[3]) -1 ;
			int	quinto_numero=cvRound(ptroRespuesta[4]) -1 ;
			int	sexto_numero=cvRound(ptroRespuesta[5]) -1 ;
            float* bit_vec = (float*)(new_responses->data.ptr + i*new_responses->step);
            for( j = 0; j < class_count; j++ )
                bit_vec[j] = 0.f;
			bit_vec[primer_numero] = 1.f;
			bit_vec[segundo_numero] = 1.f;
			bit_vec[tercer_numero] = 1.f;
		    bit_vec[cuarto_numero] = 1.f;
			bit_vec[quinto_numero] = 1.f;
			bit_vec[sexto_numero] = 1.f;
        }
		cvGetRows( data, &train_data, 0, ntrain_samples ); 
		int layer_sz[] = { data->cols,80,80, class_count };
        CvMat layer_sizes =
            cvMat( 1, (int)(sizeof(layer_sz)/sizeof(layer_sz[0])), CV_32S, layer_sz );
        mlp.create( &layer_sizes );
        printf( "Training the classifier (may take a few minutes)...\n");
	    mlp.train( &train_data, new_responses, 0, 0,
        CvANN_MLP_TrainParams(cvTermCriteria(CV_TERMCRIT_ITER,300,0.01),
#if 1
		CvANN_MLP_TrainParams::BACKPROP,0.001));
#else
        CvANN_MLP_TrainParams::RPROP,0.05));
#endif

       if( filename_to_save )
		 mlp.save( filename_to_save );
	    mlp2.load( filename_to_save );
    }
//
    mlp_response = cvCreateMat( 1, class_count, CV_32F );

 mlp_response2 = cvCreateMat( 1, class_count, CV_32F );

	time_t end;
	time (&end);
	cout<<"\nTiempo time_t: "<<(double)end-start;
	time (&start);
	vector<int>totalAciertosTest6(10,0);
	vector<int>totalAciertos6(10,0);
	vector<int>totalAciertosTest7(10,0);
	vector<int>totalAciertos7(10,0);
	vector<int>totalAciertosTest8(10,0);
	vector<int>totalAciertos8(10,0);
	vector<int>totalAciertosTest9(10,0);
	vector<int>totalAciertos9(10,0);
	vector<int>totalAciertosTest10(10,0);
	vector<int>totalAciertos10(10,0);
    vector<int>totalAciertosTest12(10,0);
    vector<int>totalAciertos12(10,0);
    vector<int>totalAciertosTest15(10,0);
    vector<int>totalAciertos15(10,0); 
	  vector<int>totalAciertos15_2(10,0);  
    vector<int>totalAciertos20(10,0);
	  vector<int>totalAciertos20_2(10,0);  
    bool crash=false;
     // compute prediction error on train and test data
    for( i = 0; i < nsamples_all; i++ )
    {
  		//redneural.data
 		crash=false;
		int best_class;
        CvMat sample;
        cvGetRow( data, &sample, i );

		vector<float>fff(120,0.0);
		for(int col=0;col<120;col++)
		{
			fff[col]=CV_MAT_ELEM(sample,float,0,col);
		}
        mlp.predict( &sample, mlp_response );
		mlp2.predict( &sample, mlp_response2 );
     
	if(1)
	{
			  CvMat* mlp_response22 = 0;
			  mlp_response22 = cvCreateMat( 1, class_count, CV_32F );
			  mlp.predict( &sample, mlp_response22 );
  			  vector<struct numProb>numProbVect22(class_count);
			  for(int col=0;col<class_count;col++)
				{
					struct numProb np;
					np.numero=col+1;
					np.prob=CV_MAT_ELEM(*mlp_response22,float,0,col);
					numProbVect22[col]=np;
				}
			 std::sort(numProbVect22.begin(),numProbVect22.end(),Compara);
			 if(numProbVect22[0].numero==1 
				   &&numProbVect22[1].numero==2 
				   &&numProbVect22[2].numero==3 
				   &&numProbVect22[3].numero==4
				   &&numProbVect22[4].numero==5)
			  {
				fdebug<<"\n i: "<<i<<"\n";
				for(int col=0;col<120;col++)
				{
					fdebug<<" "<<fff[col];
				}
				crash=true;
				numcrahses++;
		 
			 }
			 cvReleaseMat( &mlp_response22 );
   }
    vector<struct numProb>numProbVect(class_count);
 vector<struct numProb>numProbVect2(class_count);
		for(int col=0;col<class_count;col++)
		{
			struct numProb np;
			np.numero=col+1;
			np.prob=CV_MAT_ELEM(*mlp_response,float,0,col);
			numProbVect[col]=np;
		}
		std::sort(numProbVect.begin(),numProbVect.end(),Compara);


		for(int col=0;col<class_count;col++)
		{
			struct numProb np;
			np.numero=col+1;
			np.prob=CV_MAT_ELEM(*mlp_response2,float,0,col);
			numProbVect2[col]=np;
		}
		std::sort(numProbVect2.begin(),numProbVect2.end(),Compara);
		for(int col=0;col<numProbVect.size();col++)
		{
		if(	(numProbVect[col].numero)>49 ||(numProbVect[col].numero)<1)
				continue;//49 
		}
		vector<int>resultados(6,0);
		for(int col=0;col<6;col++)
			resultados[col]=(int) CV_MAT_ELEM(*responses,float,i,col);

		vector<int>resultados2(6,0);
		for(int col=0;col<6;col++)
			resultados2[col]=(int) CV_MAT_ELEM(*responses2,float,i,col);
///-------
		if(i>ntrain_samples)// && false)
	        {
		
			CvMat* ok_response = cvCreateMat( 1, class_count, CV_32F );
			int primer_numero=cvRound(resultados[0])-1 ;
			int segundo_numero=cvRound(resultados[1]) -1 ;
			int tercer_numero=cvRound(resultados[2])-1  ;
			int cuarto_numero=cvRound(resultados[3]) -1 ;
			int	quinto_numero=cvRound(resultados[4]) -1 ;
			int	sexto_numero=cvRound(resultados[5]) -1 ;

            float* bit_vec = (float*)(ok_response->data.ptr );
            for( j = 0; j < class_count; j++ )
                bit_vec[j] = 0.f;
			//cout<<"CT: "<<class_count;
           // bit_vec[cls_label] = 1.f;
			bit_vec[primer_numero] = 1.f;
			bit_vec[segundo_numero] = 1.f;
			bit_vec[tercer_numero] = 1.f;
		    bit_vec[cuarto_numero] = 1.f;
			bit_vec[quinto_numero] = 1.f;
			bit_vec[sexto_numero] = 1.f;
			 
			 
			 mlp2.train( &sample, ok_response, 0, 0,
            CvANN_MLP_TrainParams(cvTermCriteria(CV_TERMCRIT_ITER,300,0.01),
			CvANN_MLP_TrainParams::BACKPROP,0.001),1);
			// mlp.save( "RN2.data" );
    
        }
//---------
		if(i>ntrain_samples)
		{
			int aciert=0;
			int aciert2=0;
			int tamIni=6;
			while(aciert<6)
			{
				vector<int>prediccion(tamIni);//class_count,0.0);
				for(int k=0;k<tamIni;k++)
					prediccion.push_back(numProbVect[k].numero);
					vector<int>prediccion2(tamIni);//class_count,0.0);
				for(int k=0;k<tamIni;k++)
					prediccion2.push_back(numProbVect2[k].numero);
				aciert= DameCoincidencias( prediccion, resultados);
				aciert2= DameCoincidencias( prediccion2, resultados2);
				tamIni++;
				if(tamIni>=49)
					break;
			}
			fstream ff6;
			ff6.open("cantTo6.txt",ios::out);
			ff6<<(tamIni-1)<<",";
			ff6.close();
			aciert=0;
			 tamIni=6;
			 while(aciert<5)
			{
				vector<int>prediccion(tamIni);//class_count,0.0);
				for(int k=0;k<tamIni;k++)
					prediccion.push_back(numProbVect[k].numero);
				aciert= DameCoincidencias( prediccion, resultados);
				tamIni++;
				if(tamIni>=49)
					break;
			}
			fstream ff5;
			ff5.open("cantTo5.txt",ios::out);
			ff5<<(tamIni-1)<<",";
			ff5.close();
		
		}
		if( i < ntrain_samples )
			ActualizaSumas(6,numProbVect,
					resultados,totalAciertosTest6);
		else if(crash==false)
			ActualizaSumas(6,numProbVect,
				resultados,totalAciertos6);
		if( i < ntrain_samples )
			ActualizaSumas(7,numProbVect,
					resultados,totalAciertosTest7);
		else if(crash==false)
			ActualizaSumas(7,numProbVect,
				resultados,totalAciertos7);
	 		if( i < ntrain_samples )
			ActualizaSumas(8,numProbVect,
					resultados,totalAciertosTest8);
		else if(crash==false)
			ActualizaSumas(8,numProbVect,
				resultados,totalAciertos8);
					if( i < ntrain_samples )
			ActualizaSumas(10,numProbVect,
					resultados,totalAciertosTest10);
		else if(crash==false)
			ActualizaSumas(10,numProbVect,
				resultados,totalAciertos10);
			if( i < ntrain_samples )
			ActualizaSumas(12,numProbVect,
					resultados,totalAciertosTest12);
		else if(crash==false)
			ActualizaSumas(12,numProbVect,
				resultados,totalAciertos12);
			if( i < ntrain_samples )
			ActualizaSumas(15,numProbVect,
					resultados,totalAciertosTest15);
		else if(crash==false)
			ActualizaSumas(15,numProbVect,
				resultados,totalAciertos15);
				/*	if( i < ntrain_samples )
			ActualizaSumas(15,numProbVect,
					resultados,totalAciertosTest15);*/
		if( i >=ntrain_samples &&  (crash==false))
			ActualizaSumas(20,numProbVect,
				resultados,totalAciertos20);

		if( i >=ntrain_samples &&  (crash==false))
			ActualizaSumas(15,numProbVect2,
				resultados2,totalAciertos15_2);
		if( i >=ntrain_samples &&  (crash==false))
			ActualizaSumas(20,numProbVect2,
				resultados2,totalAciertos20_2);
			
 
		 
    }
	  Muestra(  totalAciertosTest6,6,true);
	  Muestra(  totalAciertos6,6,false);

	  	  Muestra(  totalAciertosTest7,7,true);
	  Muestra(  totalAciertos7,7,false);
	  Muestra(  totalAciertosTest8,8,true);
	  Muestra(  totalAciertos8,8,false);	 
	  	  Muestra(  totalAciertosTest9,9,true);
	  Muestra(  totalAciertos9,9,false);	
	  	  Muestra(  totalAciertosTest10,10,true);
	  Muestra(  totalAciertos10,10,false);	
	    	  Muestra(  totalAciertosTest12,12,true);
	  Muestra(  totalAciertos12,12,false);
	    	  Muestra(  totalAciertosTest15,15,true);
	  Muestra(  totalAciertos15,15,false);

	     Muestra(  totalAciertos15_2,15,false);
	   Muestra(  totalAciertos20,20,false);
	     Muestra(  totalAciertos20_2,20,false);
 


		//b 

   

	time (&end);
	cout<<"\nTiempo Recognition time_t: "<<(double)end-start;
	time (&start);
fdebug<<"\nnumcrahses: "<<numcrahses<<"\n";

	 

	 fdebug.close();
	// 
	
  CvMat *   mlp_response12 = cvCreateMat( 1, class_count, CV_32F );
	 CvMat * final = cvCreateMat(1, numCARACTERISTICA, CV_32F );
	 int arr[] ={
29,32,34,42,44,46,1,4,9,11,38,48,3,15,23,26,34,38,2,3,7,10,21,38,2,5,31,33,43,49,2,27,29,34,35,40,12,21,22,36,47,49,4,8,34,36,37,44,13,15,26,39,41,43,7,8,17,21,33,37,4,10,12,14,15,29,6,19,22,24,30,31,2,11,16,20,23,41,8,9,14,19,22,32,10,17,18,42,45,47,1,6,10,30,42,49,9,18,20,29,33,47,3,10,12,15,21,46,5,7,12,23,38,43,4,10,18,19,21,39
	 };
	std::vector<int> V(arr, arr + sizeof(arr)/sizeof(int));
 ///
	for(int i=0;i<V.size();i++)
	{
		CV_MAT_ELEM( *final, float, 0, i ) =V[i];
	}
 //   
    mlp.predict(  final, mlp_response12 );
 	vector<struct numProb>numProbVect(class_count);
	for(int col=0;col<class_count;col++)
	{
			struct numProb np;
			np.numero=col+1;
			np.prob=CV_MAT_ELEM(*mlp_response12,float,0,col);
			numProbVect[col]=np;
		}
	//	//FALLA
	 	std::sort(numProbVect.begin(),numProbVect.end(),Compara);
	 	ff<<" \n"<<"GGG\n ";
	 
	 	for(int k=0;k<numProbVect.size();k++)
	 		ff<<" "<<numProbVect[k].numero<<" ";

		// Save classifier to file if needed
   /* if( filename_to_save )
        mlp.save( filename_to_save );*/
ff.close();
    cvReleaseMat( &mlp_response );
    cvReleaseMat( &data );
    cvReleaseMat( &responses );

    return 0;
}
struct Sorteo
{
	int num1;
	int num2;
	int num3;
	int num4;
	int num5;
	int num6;
	Sorteo()
	{
	  num1=0;
	  num2=0;
	  num3=0;
	  num4=0;
	  num5=0;
	  num6=0;
	}
};
	

void Muestra(vector<int>totalAciertosTest,int tam,bool entreno )
{
		int sumaTest=0;
		for(int i=0;i<totalAciertosTest.size();i++)
			sumaTest+=totalAciertosTest[i];
		if(entreno)
			ff<<"\nTRAIN DATA \n";
		else
			ff<<"\nNO TRAIN \n";
		ff<<"Para "<<tam<<" numeros";
		ff<<"\n7: "<<totalAciertosTest[7]<<" 8 "<<totalAciertosTest[8]<<" 9 "<<totalAciertosTest[9] ;
		ff<<"\n6: "<<totalAciertosTest[6]<<" "<<totalAciertosTest[6]*100.0/(sumaTest*1.0);
		ff<<"\n5: "<<totalAciertosTest[5]<<" "<<totalAciertosTest[5]*100.0/(sumaTest*1.0);
		ff<<"\n4: "<<totalAciertosTest[4]<<" "<<totalAciertosTest[4]*100.0/(sumaTest*1.0); 
		ff<<"\n3: "<<totalAciertosTest[3]<<" "<<totalAciertosTest[3]*100.0/(sumaTest*1.0); 
		ff<<"\n2: "<<totalAciertosTest[2]<<" "<<totalAciertosTest[2]*100.0/(sumaTest*1.0); 
		ff<<"\n1: "<<totalAciertosTest[1]<<" "<<totalAciertosTest[1]*100.0/(sumaTest*1.0); 
		ff<<"\n0: "<<totalAciertosTest[0]<<" "<<totalAciertosTest[0]*100.0/(sumaTest*1.0); 
}
 
//http://www.terra.es/personal3/olbapordep/


int _tmain(int argc, _TCHAR* argv[])
{  
	//FILE* f = fopen( filename, "rt" );
	fstream fichero;
	//Leemos el fichero con los datos
	fichero.open(".\\bdnuevo.dat",ios::in);
	int tempActual=0;

	bool first4=true;
	int n=1;
	
 
	fstream sorteosFich;
	//fichero con el que se entrenará la red neuronal
	//resultado entradas
	sorteosFich.open("sorteosFich.txt",ios::out);
	vector<Sorteo>vectorSorteos(0);
	while(!fichero.eof())
	{
		//1.- leemos loa resultados 
		char * line=new char[256];
		fichero.getline(line,256);
		string linea=string(line);
		//-------------
		int posicion1=linea.find_first_of(",");
		int posicion2=linea.find_first_of(",",posicion1+1);
		Sorteo sorteo;
		string number1="";
		//1
		for(int i=posicion1+1;i<posicion2;i++)
		{
			number1.push_back(linea[i]);
			sorteo.num1 =atoi(number1.c_str());
		}
		posicion1=posicion2 ;
		posicion2=linea.find_first_of(",",posicion1+1);
		number1="";
		//2
		for(int i=posicion1+1;i<posicion2;i++)
		{
			number1.push_back(linea[i]);
			sorteo.num2 =atoi(number1.c_str());
		}
		posicion1=posicion2 ;
		posicion2=linea.find_first_of(",",posicion1+1);
		number1="";
		//3
		for(int i=posicion1+1;i<posicion2;i++)
		{
			number1.push_back(linea[i]);
			sorteo.num3 =atoi(number1.c_str());
		}
		posicion1=posicion2 ;
		posicion2=linea.find_first_of(",",posicion1+1);
		number1="";
		//4
		for(int i=posicion1+1;i<posicion2;i++)
		{
			number1.push_back(linea[i]);
			sorteo.num4 =atoi(number1.c_str());
		}
			posicion1=posicion2 ;
		posicion2=linea.find_first_of(",",posicion1+1);
		number1="";
		//5
		for(int i=posicion1+1;i<posicion2;i++)
		{
			number1.push_back(linea[i]);
			sorteo.num5 =atoi(number1.c_str());
		}
			posicion1=posicion2 ;
		posicion2=linea.find_first_of(",",posicion1+1);
		number1="";
		//6
		for(int i=posicion1+1;i<posicion2;i++)
		{
			number1.push_back(linea[i]);
			sorteo.num6 =atoi(number1.c_str());
		}	 
		//---------------
    	vectorSorteos.push_back(sorteo);
 		sorteosFich<<sorteo.num1<<" ";
		sorteosFich<<sorteo.num2<<" ";
		sorteosFich<<sorteo.num3<<" ";
		sorteosFich<<sorteo.num4<<" ";
		sorteosFich<<sorteo.num5<<" ";
		sorteosFich<<sorteo.num6<<" ";
		sorteosFich<<"\n";
	}	

	cout<<"READ\n\a";
	sorteosFich.close();
	//getchar();	
				fstream rnfich;

				//fichero con el que se entrenará la red neuronal
				//resultado entradas
				rnfich.open("rnfich.txt",ios::out);
vector<Sorteo>vectorSorteosAlReves(0);
for(int i=vectorSorteos.size()-1;i>=0;i--)
{
	vectorSorteosAlReves.push_back(vectorSorteos[i]);
}
	//vectorSorteos[vectorSorteos.size()-1]
	int numS= numCARACTERISTICA/6;
	int k=0;
	for(int i=1+numS;i<vectorSorteosAlReves.size();i++)
	{
		k++;
		Sorteo result=vectorSorteosAlReves[i];//11
		rnfich<<result.num1<<",";	
		rnfich<<result.num2<<",";	
		rnfich<<result.num3<<",";	
		rnfich<<result.num4<<",";	
		rnfich<<result.num5<<",";	
		rnfich<<result.num6<<",";	
		for (int j=0;j<numS;j++)
		{
			Sorteo otros=vectorSorteosAlReves[i-(j+1)];
			rnfich<<otros.num1<<",";	
			rnfich<<otros.num2<<",";	
			rnfich<<otros.num3<<",";	
			rnfich<<otros.num4<<",";	
			rnfich<<otros.num5<<",";	
			rnfich<<otros.num6;
			if(j<numS-1)
				rnfich<<",";
		}
		rnfich<<"\n";	
	}
	rnfich.close();
				//rnfich<<"\n"<<resultado<<",";	rnfich<<clubLocal->golesAFavor;

				
				

	CvANN_MLP mlp;

	int DifChars=49;
	CvMat * data = cvCreateMat(k, numCARACTERISTICA, CV_32F );
	//CV_MAT_ELEM( *data,short, 0, 0 ) = (short)33;
	//CV_MAT_ELEM( *image_points, float, i, 1 ) = corners[j].y;

	CvMat * responses = cvCreateMat( k, 6, CV_32F );
	 k=0;
	for(int i=1+numS;i<vectorSorteosAlReves.size();i++)
	{
		 
		Sorteo result=vectorSorteosAlReves[i];//11
 
		CV_MAT_ELEM( *responses, float, k, 0 ) =1.0*result.num1;
		CV_MAT_ELEM( *responses, float, k, 1 ) =1.0*result.num2;
		CV_MAT_ELEM( *responses, float, k, 2 ) =1.0*result.num3;
		CV_MAT_ELEM( *responses, float, k, 3 ) =1.0*result.num4;
		CV_MAT_ELEM( *responses, float, k, 4 ) =1.0*result.num5;
		CV_MAT_ELEM( *responses, float, k, 5 ) =1.0*result.num6;
		for (int j=0;j<numS;j++)
		{
			Sorteo otros=vectorSorteosAlReves[i-(j+1)];
			CV_MAT_ELEM( *data, float, k, j*6 ) =1.0*otros.num1;
 			CV_MAT_ELEM( *data, float, k, j*6+1 ) =1.0*otros.num2;
			CV_MAT_ELEM( *data, float, k, j*6+2 ) =1.0*otros.num3;
			CV_MAT_ELEM( *data, float, k, j*6+3 ) =1.0*otros.num4;
			CV_MAT_ELEM( *data, float, k, j*6+4 ) =1.0*otros.num5;
			CV_MAT_ELEM( *data, float, k, j*6+5 ) =1.0*otros.num6;
		 
		}
	 
		k++;
	}
	//rnfich fichero resultado,entradas
	//numCARACTERISTICA= numero de entradas
//	for(int i=0;i<responses->rows;i++)
//{
//	for(int j=0;j<responses->cols;j++)
//	{
//		float* s=( float*)(responses->data.fl +(i*responses->cols+j));
//		//cout<<s[0]<<" ";
//	}
//	//cout<<"\n";
//}
//for(int i=0;i<data->rows;i++)
//{
//	for(int j=0;j<data->cols;j++)
//	{
//		float *s=(float*)(data->data.fl+(i*data->cols +j ));
//		cout<<s[0]<<" ";
//	}
//	cout<<"\n";
//}
	//Difchars = num de de diferentes salidas
//build_mlp_classifier2( char* data_filename,
//    char* filename_to_save, char* filename_to_load,CvANN_MLP mlp,int numCARACTERISTICAS,int DifChars ,CvMat* data,

#define TRAINING
#ifdef TRAINING
	build_mlp_classifier2("rnfich.txt","redneural.data",
					 0,mlp, numCARACTERISTICA,DifChars,data,responses );
#else
	cout<<"entering build_mlp!;";
	build_mlp_classifier2("rnfich.txt",0,
					 "redneural.data",mlp, numCARACTERISTICA,DifChars,data,responses );
#endif
	 
	//getchar();
	cout<<"\a\a";
	return 0;
}

