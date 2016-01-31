package sad.mains;

import java.text.SimpleDateFormat;
import java.util.Calendar;
import java.util.GregorianCalendar;

import sad.datahandlers.AdHocRanking;
import sad.datahandlers.DataLoader;
import sad.evaluation.GridSearchWithCVParam;
import sad.evaluation.Multibounds;
import sad.evaluation.Preprocess;
import sad.utils.Stopwatch;
import sad.utils.VerboseCutter;
import weka.classifiers.functions.LibSVM;
import weka.core.Instances;
import weka.core.SelectedTag;

/**
 * 
 * @author Iñigo Sánchez Méndez @espolex
 *  
 */
public class ScanParamsAlgorithm 
{
	/**
	 * Primer argumento es el path del archivo arff con el conjunto de instancias de entrenamiento.
	 * El segundo argumento puede ser el path del archivo arff con el conjunto de instancias a predecir (test).
	 * Tercer argumento sirve para diferentes opciones descritas en el Readme.txt.	 
	 */
	public static void main(String[] args) throws Exception 
	{				
		// Comprobamos si el número de argumentos es el correcto.
		if (args.length < 1 || args.length>3)
		{
			printError();
			return; 					
		}			
		// Comprobación del path y extensión de archivo de entrenamiento y test.
		Instances trainData = null;
		Instances testData = null;
		DataLoader trainLoader = null;
		DataLoader testLoader = null;
		
		//Instanciamos variables boolean dependientes de los argumentos:
		//Para los filtros
		boolean fNormalize=false;
		boolean fBalance=true;
		boolean fExtremeOutliers=false;
		
		//Para establecer método de evaluación (no honesto/5 fold cross)
		boolean foldC = true;
		
		//Para obtener resultados de la optimización de parámetros del modelo con GridSearch
		boolean GS=false;
		
		//Para obtener baseline
		boolean baseline=false;
		
		//Para saber si se envía por parámetros el archivo arff de las instancias de test.
		boolean hayTest=true;
		
		if(args.length==1)
		{
			trainLoader = new DataLoader(args[0]);
			Instances allData = trainLoader.instancesLoader();
			Instances[] partitions = Preprocess.getMiPreprocess().doPartition(allData);
			hayTest = false;
			try
			{
				// Se cargan las instancias, se barajan y se dispone como clase el último atributo.
				 trainData = partitions[0];							 
			
				//Se cargan las instancias a predecir de la misma forma que el conjunto de entrenamiento				
				 testData = partitions[1];
			}
			catch(NullPointerException e)
			{
				System.out.println("Imposible particionado");
				printError();
			}
		}
		else if(args.length==2)
		{
			if(args[1].contains(".arff"))
			{
				trainLoader = new DataLoader(args[0]);
			    trainData = trainLoader.instancesLoader();
			    testLoader = new DataLoader(args[1]);
			    testData= testLoader.instancesLoader();
			}
			else
			{
				hayTest = false;
				trainLoader = new DataLoader(args[0]);
				Instances allData = trainLoader.instancesLoader();
				Instances[] partitions = Preprocess.getMiPreprocess().doPartition(allData);
				try
				{
					// Se cargan las instancias, se barajan y se dispone como clase el último atributo.
					 trainData = partitions[0];							 
				
					//Se cargan las instancias a predecir de la misma forma que el conjunto de entrenamiento				
					 testData = partitions[1];
				}
				catch(NullPointerException e)
				{
					System.out.println("Imposible particionado");
					printError();
				}
				if(args[1].contains("D"))
				{
					//Metodo deshonesto
					foldC=false;
				}
				if(args[1].contains("B"))
				{
					//OneR
					baseline=true;
				}
				if(args[1].contains("R"))
				{
					//Activa filtro RSAMPLE
					fBalance=true;
				}
				if(args[1].contains("O"))
				{
					//Desactiva filtro para eliminar outliers y extreme values.
					fExtremeOutliers=false;
				}
				if(args[1].contains("G"))
				{
					//Activa gridsearch
					GS=true;
				}
			}
		}
		else if (args.length==3)
		{
			trainLoader = new DataLoader(args[0]);
		    trainData = trainLoader.instancesLoader();
		    testLoader = new DataLoader(args[1]);
		    testData= testLoader.instancesLoader();
		    if(args[2].contains("D"))
			{
				//Metodo deshonesto
				foldC=false;
			}
			if(args[2].contains("B"))
			{
				//Activa resultados oneR.
				baseline=true;
			}
			if(args[2].contains("R"))
			{
				//Activa filtro resample.
				fBalance=true;
			}
			if(args[2].contains("O"))
			{
				//Desactiva filtro para eliminar outliers y extremevalues.
				fExtremeOutliers=false;
			}
			if(args[2].contains("G"))
			{
				GS=true;
			}
		}	
		
		// Separador para el ranking de resultados.
		String separador="====";
		for (int s = 0; s < 30; s++)
		{
			separador = separador.concat("====");
		}
				
		// Guardamos la fecha y hora actuales para ponerle el nombre al fichero de resultados.
		Calendar calendar = new GregorianCalendar(); // Fecha y hora actuales.
		SimpleDateFormat dateFormat = new SimpleDateFormat("yyyyMMdd-HHmm"); // Formato de la fecha.
		String dateS = dateFormat.format(calendar.getTime()); // Fecha y hora actuales formateadas.
		
		// Variable para captura ranking y variable para guardar el contenido en un archivo.
		String rankingPath = args[0].substring(0, args[0].length() - 5) + "-ranking-" + dateS + ".txt";
		String ranking;
				
		// Variable para la captura de estadísticas.
		String summary = new String();
		
		// Variables para comparar la f-measure en curso con la f-measure de la vuelta anterior.
		double fmeasureAux = 0.0;
		double fmeasureBest = 0.0;
		
		// Variable para imprimir el tiempo de ejecución de las predicciones por consola.
		double elapsedTime = 0.0;
				
		// Variable para obtener la mejor C.
		double bestC = 0;
		//Variable para obtener el mejor gamma.
		double bestG = 0;
		
	    // Utilizamos el kernel por defecto RBF
		//Inicio de barrido de parámetros
		int maxOfCSearch = 11; //Hasta  que valor es óptimo C "barrer"??
		int maxOfGSearch = 4;  //Hasta  que valor es óptimo					

		if(fExtremeOutliers)
		{
			//Filtro para eliminar ouliers y extreme 
			try 
			{
				int numAtt = trainData.numAttributes();
				trainData = Preprocess.getMiPreprocess().getFilterInstancesWithoutOutliers(trainData);
				if(numAtt!=trainData.numAttributes())throw new Exception();
				
			} 
			catch (Exception e) 
			{
				System.out.println("Error al intentar eliminar extreme values y outliers");
				return;
			}
		}
		if(fNormalize)
		{
			//Normalizamos los valores que pueden tomar los atributos ( [-1,1] )
			try 
			{
				trainData = Preprocess.getMiPreprocess().getFilterInstancesWithNormalize(trainData,-1.1);
				testData = Preprocess.getMiPreprocess().getFilterInstancesWithNormalize(testData, -1.1);
			} 
			catch (Exception e) 
			{
				
				System.out.println("Error al intentar normalizar");
			}
		}		
		if(fBalance)
		{
			//Filtros para balancear y evitar overfit
			try {
				trainData = Preprocess.getMiPreprocess().getBalancedInstances(trainData);
			} 
			catch (Exception e) 
			{				
				System.out.println("Error al intentar balancear las instancias");;
			}
		}
		// Creamos una instancia del clasificador y el evaluador a usar.
		LibSVM estimador = new LibSVM();
		
		//Utilizamos C-SVC
		estimador.setSVMType(new SelectedTag(0, LibSVM.TAGS_SVMTYPE));
		
		//Establecemos el kernel RBF justiificado en el guión
		estimador.setKernelType(new SelectedTag(2,LibSVM.TAGS_KERNELTYPE));	
		
		Multibounds evaluator = new Multibounds(trainData);
		
		//barrido del parámetro c = 2^C, justificado en el guión porqué se utilizan potencias de 2
		//Lo activamos para evitar el modo verbose de libSVM 
		System.out.println("Starting C and Gamma optimization with svm model");
		VerboseCutter.getVerboseCutter().cutVerbose();
		
		for(int c = -15;c <=maxOfCSearch; c++)
		{			
			for (int g = -3; g <= maxOfGSearch; g++)
			{
				estimador = new LibSVM();
				double cost = (Math.pow(2, c));
				double gamma = (Math.pow(2, g));
				estimador.setGamma(gamma);
				estimador.setCost(cost);
				
				evaluator = new Multibounds(trainData);
				try
				{
					if (foldC)//foldC== true -> 5 fold cross
					{	
						//Lo activamos para evitar el modo verbose de libSVM 
						//VerboseCutter.getVerboseCutter().cutVerbose();
						evaluator.assesPerformanceNFCV(estimador, 5, trainData);
						//Si no lo reactivamos dejan de funcionar las salidas por pantalla 
						//VerboseCutter.getVerboseCutter().activateVerbose();
					}
					else //foldC== false -> evaluacion deshonesta
					{
						evaluator.dishonestEvaluator(estimador, trainData);										
					}
				}
				catch(Exception e)
				{
					evaluator = new Multibounds(trainData);
				}	
				
				fmeasureAux=evaluator.weightedFMeasure();

				// Comprobamos la mejor f-measure.
				if (fmeasureAux > fmeasureBest)
				{
					fmeasureBest = fmeasureAux;
					bestC = cost;
					bestG = gamma;
				    summary = "Tiempo de ejecución : " + elapsedTime + "\n" + evaluator.toSummaryString() + 
							"Recall:\t " + evaluator.weightedRecall() + "\nPrecision:\t " + 
							evaluator.weightedPrecision() + "\n" + evaluator.toMatrixString()
							+"\n" +"weightedROC : " + evaluator.weightedAreaUnderROC()
							+"\n" +"Fmeasure:	"+ evaluator.weightedFMeasure()+
							"\n Mejor C: "+bestC+
							"\n Mejor gamma: "+bestG;
					
					// Guardamos los resultados en el fichero del ranking.
					trainLoader.SaveFile(rankingPath, separador, false);
					AdHocRanking adHocRanking = new AdHocRanking( bestC, bestG, summary);
					ranking=adHocRanking.toStringSVM(fmeasureBest);
					trainLoader.SaveFile(rankingPath, ranking, false);
				}
			}
		}
		//Redirigimos la salidas.
		VerboseCutter.getVerboseCutter().activateVerbose();
		System.out.println("Finished C and Gamma optimization with svm model");

		//Mostramos las estadísticas del mejor clasificador.	
		System.out.println(summary);
		
		//Para medir el tiempo que tarda en predecir
		Stopwatch predictionTime = new Stopwatch();
		//Predecir la clase estimada de cada instancia
		
		Instances labeled = evaluator.predictionsMaker(estimador, testData);
		elapsedTime = predictionTime.elapsedTime();
		
		System.out.println("La predicción del conjunto de datos estimadas por el modelo "+estimador.getClass().toString()+
				"\n Le ha llevado un tiempo de ejecución: "+elapsedTime);
		
		//Guardar las clases estimadas para cada instancia en formato .arff	
		
		if(hayTest)
		{
		testLoader = new DataLoader(args[1]);
		testLoader.pushDataSets(labeled, "EstimadasSVM",args[1]);
		}
		else
		{
			trainLoader.pushDataSets(labeled, "EstimadasSVM", args[0]);
		}		 
		if(baseline)
		{
			System.out.println("----------------Baseline-----------------");
			ScanParamsOneR.main(args);
		}
		
		if(GS)
		{
			System.out.println("----------------GridSearch-----------------");
			GridSearchWithCVParam.getGSParams(trainData).getBestParamsSVM(trainData, foldC);
		}
	}
	/**
	 * pre:
	 * post: Escribe el objetivo del sistema y el orden y significado de los parámetros por consola.
	 */
	private static void printError() 
	{
		System.out.println("OBJETIVO: Buscar parámetros  óptimos para el clasificador c-SVM con las instancias dadas, "
		+ " evaluando mediante 5-fold cross-validation.");
		System.out.println("ARGUMENTOS, se debe respetar el orden de precedencia:");
		System.out.println("\t1. Path del fichero de entrenamiento: datos en formato .arff");
		System.out.println("\t2. Path del fichero de test: datos en formato .arff");
		System.out.println("\t3. -F Posibilidad de filtrar o no el conjunto de instancias a traves "
				+ "de los distintos filtros, por defecto se aplican todos los filtros");
		System.out.println("\t4. Numérico entre tal-cual para decidir que filtros utilizar");
		System.out.println("\t5. -BS además obtiene los resultados del modelo baseline OneR, por defecto no lo obtiene");
		//System.out.println("\t6. -NH además obtiene los resultados obtenidos con el método no honesto, por defecto no lo obtiene");
		//System.out.println("\t7. -GS obtener los resultados de la optimización GridSearch");
	}
}
