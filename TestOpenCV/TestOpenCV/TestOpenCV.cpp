// TestOpenCV.cpp: define el punto de entrada de la aplicación de consola.
//

#include "stdafx.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <string.h>

using namespace cv;


void SuavizaCurva(vector<int>&suma_negros_columnas);
vector<int>  DameHistogramaHorizontal(const Mat & gray);
vector<int>  DameHistogramaVertical(const Mat & gray);
int DamePosicionDelMaximo(vector<int> suma_negros_columnas);
int DamePuntoSuperiorMaximo(vector<int> suma_negros_columnas, int umbral);
int DamePuntoInferiorMaximo(vector<int> suma_negros_columnas, int umbral);


int main( int argc, char** argv )
{
	const string nombre_fichero_imagen="imagen.jpg";
	//El vector de puntos será el resultado final que queremos alcanzar,
	//es decir, el conjunto de puntos que conforman el borde de la pupila
	vector<cv::Point> vector_puntos(0);

	// imagen_color, contendrá la imagen original a tratar
	Mat imagen_color,copia_original;
	Mat imagen_temporal_color;
	// imagen_color, contendrá la imagen original a tratar
	Mat image_gris;
	Mat  matriz_deteccion, recorte_color;
	//Estos vectores contendrán una proyección sobre el eje de la cantidad de pixels negros
	vector<int> suma_negros_columnas, suma_negros_filas;
	
	int pos_max_columnas;
	int pos_max_filas;
	int umbral_columnas,x_0,x_1;
	int umbral_filas,y_0,y_1;
	   
	//Leemos el archivo 
    imagen_color = imread(nombre_fichero_imagen, IMREAD_COLOR); 
    copia_original= imagen_color.clone();
	
	if(! imagen_color.data ) 
    {
        std::cout << "Error al abrir el fichero " << std::endl ;
        return -1;
    }
	
	//Creamos una ventana para mostrar la imagen original
    namedWindow( "Imagen Original",cv::WINDOW_NORMAL );
	//Mostramos la imagen original
    imshow( "Imagen Original", imagen_color ); 

	//Difuminamos para quitar ruido
	GaussianBlur( imagen_color, imagen_color, Size( 3, 3 ), 0, 0 );
	
	//Transformamos a escala de image_grises
	cv::cvtColor(imagen_color, image_gris, CV_BGR2GRAY);

    namedWindow( "Imagen en gris",cv::WINDOW_NORMAL ); 
    imshow     ( "Imagen en gris", image_gris );  
	 
	
	//Aplicamos el umbral para transformar a blanco y negro
	cv::threshold(image_gris,image_gris,0.4*sum(image_gris)[0]/(image_gris.cols*image_gris.rows),255,cv::THRESH_BINARY );
   	namedWindow( "Blanco y Negro",cv::WINDOW_NORMAL );
    imshow( "Blanco y Negro", image_gris ); 
 
	cv::cvtColor( image_gris, imagen_temporal_color, CV_GRAY2RGB); 
	suma_negros_columnas=DameHistogramaHorizontal(image_gris);//.cols,0 );

	suma_negros_filas=DameHistogramaVertical(image_gris);//.cols,0 );
		
	
 
	pos_max_columnas=DamePosicionDelMaximo(suma_negros_columnas);
	umbral_columnas=(0.15* suma_negros_columnas[pos_max_columnas]);

	x_1=DamePuntoInferiorMaximo(suma_negros_columnas,umbral_columnas);;
	x_0=DamePuntoSuperiorMaximo(suma_negros_columnas,umbral_columnas);

	pos_max_filas=DamePosicionDelMaximo(suma_negros_filas);
    umbral_filas=(0.15* suma_negros_filas[pos_max_filas]);

	y_1=DamePuntoInferiorMaximo(suma_negros_filas,umbral_filas);;
	y_0=DamePuntoSuperiorMaximo(suma_negros_filas,umbral_filas);
	
	//Mostramos la proyección de pixels 
	for(int i=0;i<image_gris.cols;i++)
	{ 
		cv::line(imagen_temporal_color,cv::Point(i,0),cv::Point(i,suma_negros_columnas[i]),Scalar(0,0,255), 3, 8, 0 );
	}

 	cv::line(imagen_temporal_color,cv::Point(0,y_0),cv::Point(image_gris.cols,y_0 ),Scalar(255,0,255), 2, 8, 0 );
	cv::line(imagen_temporal_color,cv::Point(0,y_1),cv::Point(image_gris.cols,y_1 ),Scalar(255,0,255), 2, 8, 0 );


	for(int i=0;i<image_gris.rows;i++)
	{ 
		cv::line(imagen_temporal_color,cv::Point(0,i),cv::Point(suma_negros_filas[i],i),Scalar(255,0,0), 3, 8, 0 );
	}

	cv::line(imagen_temporal_color,cv::Point(x_0,0),cv::Point(x_0, image_gris.rows),Scalar(0,255,255), 2, 8, 0 );
		
	cv::line(imagen_temporal_color,cv::Point(x_1,0),cv::Point(x_1, image_gris.rows),Scalar(0,255,255), 2, 8, 0 );

	
	namedWindow( "gráfica",cv::WINDOW_NORMAL );
    imshow( "gráfica", imagen_temporal_color );
	
	//A la hora de recortar la imagen aplicaremos un 20% de desplazamiento, a modo de margen de error.
	int dx=(x_1-x_0)/10;
	int dy=(y_1-y_0)/10;
 
	recorte_color=copia_original(cv::Rect(x_0 -dx   ,y_0-dy , (x_1-x_0 +2*dx) ,(y_1-y_0 +2*dy)  ) ).clone();
	
	namedWindow( "recorte_color",cv::WINDOW_NORMAL );
    imshow( "recorte_color", recorte_color ); 
	
	Mat recorte_gris;
	cv::cvtColor(recorte_color,recorte_gris, CV_BGR2GRAY);
	
	namedWindow( "recorte_gris",cv::WINDOW_NORMAL );
    imshow( "recorte_gris", recorte_gris );  
	
	//Difuminamos la imagen
    blur( recorte_gris, matriz_deteccion, Size(3,3) );
	//Aplicamos el algoritmo de detección de bordes de Canny
	//http://es.wikipedia.org/wiki/Algoritmo_de_Canny
   
	Canny( matriz_deteccion, matriz_deteccion, 30, 180, 3 );
    
	//Invertimos los colores
	bitwise_not(matriz_deteccion,matriz_deteccion);
	//Aplicamos una erosión 
	cv::erode(matriz_deteccion,matriz_deteccion,Mat());

    namedWindow( "Canny", WINDOW_NORMAL );
	imshow( "Canny", matriz_deteccion );
  
	// Mostramos los bordes detectados en el recorte
    Mat  dst =   Mat(recorte_color.size(),recorte_color.type(),Scalar::all(255));//.clone();
    recorte_color.copyTo( dst, matriz_deteccion);
  
	namedWindow( "detección", WINDOW_NORMAL );
	imshow( "detección", dst );
  
	//Llenamos el vector de puntos que estabamos buscando
	for(int i=0;i<matriz_deteccion.cols ;i++)
	{
		for(int j=0;j<matriz_deteccion.rows;j++)
		{
			//Hay que tener en cuenta que los puntos detectados están en el recorte
			//Por eso, habrá que sumarles los desplazamientos
			if(matriz_deteccion.at<uchar>(cv::Point(i,j))==0)
				vector_puntos.push_back(cv::Point(i+x_0 -dx ,j+y_0-dy ));
		}
	} 
	//Mostramos el resultado
	for(int i=0;i<vector_puntos.size() ;i++)
	{
		cv::circle(copia_original,vector_puntos[i],0.1,cv::Scalar(255,0,0),0.1);
	} 
	namedWindow( "resultado", WINDOW_NORMAL );
	imshow( "resultado", copia_original );

    //Guardamos el resultado
	cv::imwrite("resultado.jpg",copia_original);
  	// Esperamos la pulsación de una tecla
    waitKey(0); 
	return 0;
   
}

vector<int>  DameHistogramaVertical(const Mat & gray)
{
	vector<int> suma_negros_filas(gray.rows,0 );

	for(int j=gray.rows/10;j<gray.rows*0.9;j++)
	{
			for(int i=0;i<gray.cols;i++)
			{
		
				suma_negros_filas[j]+=gray.at<uchar>(cv::Point(i,j))>0?0:1;
		
				}
	 

	}
	SuavizaCurva(suma_negros_filas);
	return suma_negros_filas;
}

vector<int>  DameHistogramaHorizontal(const Mat & gray)
{
	vector<int> suma_negros_columnas(gray.cols,0 );

	for(int i=gray.cols/10;i<gray.cols*0.9;i++)
	{
		for(int j=0;j<gray.rows;j++)
		{

				suma_negros_columnas[i]+=gray.at<uchar>(cv::Point(i,j))>0?0:1;
		
		}
	 

	}
	SuavizaCurva(suma_negros_columnas);
	return suma_negros_columnas;
}

void SuavizaCurva(vector<int>&suma_negros_columnas){
		vector<int> suma_negros_columnastemp(suma_negros_columnas.size() ,0 );
		for(int i=2;i<	suma_negros_columnas.size()-2;i++)
		{
			suma_negros_columnastemp[i]= (suma_negros_columnas[i-2]+suma_negros_columnas[i-1]+suma_negros_columnas[i]+suma_negros_columnas[i+1]+suma_negros_columnas[i+2])/5;
		}
		suma_negros_columnas= suma_negros_columnastemp;
		 
}

int DamePosicionDelMaximo(vector<int> suma_negros_columnas)
{
int max=0;
		int pos_max=0;
		for(int i=1;i<	suma_negros_columnas.size()-1;i++)
		{
			if(suma_negros_columnas[i]>max){
			max=suma_negros_columnas[i]; pos_max=i;
			}
		}
		return pos_max;

}
int DamePuntoSuperiorMaximo(vector<int> suma_negros_columnas, int umbral){
	
	int pos_max=DamePosicionDelMaximo(suma_negros_columnas);
	int pos_max_1=  pos_max;
		for(int i=pos_max;i>0 ;i--)
		{
			if(suma_negros_columnas[i]< 0.5* suma_negros_columnas[pos_max]){
				pos_max_1=i;
			break;
			}
		}
		return pos_max_1;
}
int DamePuntoInferiorMaximo(vector<int> suma_negros_columnas, int umbral){
	
	int pos_max=DamePosicionDelMaximo(suma_negros_columnas);
	int pos_max_2=  pos_max;
	for(int i=pos_max;i<	suma_negros_columnas.size()-1;i++)
		{
			if(suma_negros_columnas[i]< umbral){
				pos_max_2=i;
			break;
			}
		}
		return pos_max_2;
}