package sad.mains;

import java.text.SimpleDateFormat;
import java.util.Calendar;
import java.util.GregorianCalendar;

import sad.datahandlers.AdHocRanking;
import sad.datahandlers.DataLoader;
import sad.evaluation.Multibounds;
import sad.evaluation.Preprocess;
import sad.evaluation.ScanCVParams;
import sad.utils.Stopwatch;
import sad.utils.VerboseCutter;
import weka.classifiers.rules.OneR;
import weka.core.Instances;

public class ScanParamsOneR
{
	/**
	 * Primer argumento es el path del archivo arff con el conjunto de instancias de entrenamiento.
	 * El segundo argumento puede ser el path del archivo arff con el conjunto de instancias a predecir (test).
	 * Tercer argumento sirve para diferentes opciones descritas en el Readme.txt.	 
	 * @param args
	 * @throws Exception
	 */
	public static void main(String[] args) throws Exception 
	{
		//1
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
		boolean fNormalize=true;
		boolean fBalance=true;
		boolean fExtremeOutliers=true;
		
		//Para establecer método de evaluación (no honesto/5 fold cross)
		boolean foldC = true;
		
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
			    testData = testLoader.instancesLoader();
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
				if(args[1].contains("R"))
				{
					fBalance=false;
				}
				if(args[1].contains("O"))
				{
					fExtremeOutliers=false;
				}
			}
		}
		else 
		{
			trainLoader = new DataLoader(args[0]);
		    trainData = trainLoader.instancesLoader();
		    testLoader = new DataLoader(args[1]);
		    testData = testLoader.instancesLoader();
		    if(args[2].contains("D"))
			{
				//Metodo deshonesto
				foldC=false;
			}
			if(args[2].contains("R"))
			{
				//Desactiva filtro resample.
				fBalance=false;
			}
			if(args[2].contains("O"))
			{
				//Desactiva filtro para eliminar outliers y extremevalues.
				fExtremeOutliers=false;
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
		SimpleDateFormat dateFormat = new SimpleDateFormat("yyyyMMdd-HHmmss"); // Formato de la fecha.
		String dateS = dateFormat.format(calendar.getTime()); // Fecha y hora actuales formateadas.
		

		if(fExtremeOutliers)
		{	
			//Filtro para eliminar ouliers y extreme 
			try {
				int numAtt = trainData.numAttributes();
				trainData = Preprocess.getMiPreprocess().getFilterInstancesWithoutOutliers(trainData);
				if(numAtt!=trainData.numAttributes())throw new Exception();
				
			} catch (Exception e) {
				System.out.println("Error al intentar eliminar extreme values y outliers");
				return;
			}
		}
		if(fNormalize)
		{
			//Normalizamos los valores que pueden tomar los atributos ( [-1,1] )
			try {
				trainData = Preprocess.getMiPreprocess().getFilterInstancesWithNormalize(trainData,-1.1);
				testData = Preprocess.getMiPreprocess().getFilterInstancesWithNormalize(testData, -1.1);
			} catch (Exception e) {
				System.out.println("Error al intentar normalizar");
			}
		}		
		if(fBalance)
		{
			//Filtros para balancear y evitar overfit
			try {
				trainData = Preprocess.getMiPreprocess().getBalancedInstances(trainData);
			} catch (Exception e) 
			{
				System.out.println("Error al intentar balancear las instancias");;
			}
		}
		
		// Variable para captura ranking y variable para guardar el contenido en un archivo.
		String rankingPath = args[0].substring(0, args[0].length() - 5) + "-ranking-" + dateS + ".txt";
		String ranking;
		
		// Variable para la captura de estadísticas.
		String summary = new String();
		
		// Variables para comparar la f-measure en curso con la f-measure de la vuelta anterior.
		double fmeasureAux = 0.0;
		double fmeasureBest = 0.0;
		
		// Variable para obtener la mejor B.
		int bestB = 0;
		
		// Creamos una instancia del clasificador a usar.
		OneR oR= new OneR();
		
		//Evaluación 5FCV
		Multibounds evaluator = new Multibounds(trainData);
		System.out.println("Starting B optimization with OneR and ad hoc method");
		VerboseCutter.getVerboseCutter().cutVerbose();
		// Inicio del barrido de parámetros.
		for(int B = 1; B < trainData.numInstances(); B++)
		{
			oR.setMinBucketSize(B);
			evaluator = new Multibounds(trainData);
			if (foldC)
			{//Evaluación con el método 5FCV
			try 
			{
				evaluator.assesPerformanceNFCV(oR, 5, trainData);
			} 
			catch (Exception e) 
			{
				evaluator = new Multibounds(trainData);
			}
			}
			else
			{//Evalucio deshonesta
				evaluator.dishonestEvaluator(oR, trainData);
			}
			
			
			// Obtenemos la f-measure de la vuelta actual del bucle.
			fmeasureAux = evaluator.weightedFMeasure();
			
			// Comprobamos la mejor f-measure.
			if (fmeasureAux > fmeasureBest)
			{
				fmeasureBest = fmeasureAux;
				bestB = B;
				summary =  "Fin del barrido de parámetros\n"+evaluator.toSummaryString() + 
						"Recall:\t " + evaluator.weightedRecall() + "\nPrecision:\t " + 
						evaluator.weightedPrecision() + "\n" + evaluator.toMatrixString()
						+"\n" +"weitheredROC : " + evaluator.weightedAreaUnderROC() + "\n con F-measure:" + 
						evaluator.weightedFMeasure();
				
				// Guardamos los resultados en el fichero del ranking.
				trainLoader.SaveFile(rankingPath, separador, false);
				AdHocRanking adHocRanking = new AdHocRanking(bestB,summary);
				ranking=adHocRanking.toStringOneR(fmeasureBest);
				trainLoader.SaveFile(rankingPath, ranking, false);
			}
		}
		VerboseCutter.getVerboseCutter().activateVerbose();
		System.out.println("Finished B optimization with OneR and ad hoc method");

		
		// Mostramos las estadísticas del mejor clasificador.	
		System.out.println(summary);
		
	    //Para medir el tiempo que tarda en predecir
		Stopwatch predictionTime = new Stopwatch();
		//Predecir la clase estimada de cada instancia
		Instances labeled = evaluator.predictionsMakerGeneric(oR, testData);
		double elapsedTime = predictionTime.elapsedTime();
		System.out.println("La mejor B :\t"+ bestB);
		System.out.println("La predicción del conjunto de datos estimadas por el modelo "+oR.getClass().toString()+
				"\n Le ha llevado un tiempo de ejecución: "+elapsedTime);
		
		//Guardar las clases estimadas para cada instancia en formato .arff	
		
		if(hayTest)
		{
			testLoader = new DataLoader(args[1]);
			testLoader.pushDataSets(labeled, "EstimadasOneR",args[1]);
		}
		else
		{
			trainLoader.pushDataSets(labeled, "EstimadasOneR", args[0]);
		}
				
		 ScanCVParams.getScanCVParams(trainData).getBestParamsOneR(trainData, foldC);
	}
	private static void printError() {
		System.out.println("OBJETIVO: Buscar parámetro B óptimo para el clasificador OneRule con las instancias dadas, "
		+ " evaluando mediante 5-fold cross-validation.");
		System.out.println("ARGUMENTOS, se debe respetar el orden de precedencia:");
		System.out.println("\t1. Path del fichero de entrenamiento: datos en formato .arff");
		System.out.println("\t2. Path del fichero de test: datos en formato .arff");
		System.out.println("\t3. Para elegir el tipo de evaluación, obligatorio, 0 NO HONESTO y 1 5FCV");
		System.out.println("\t4. -F Posibilidad de filtrar o no el conjunto de instancias a traves "
				+ "de los distintos filtros, por defecto se aplican todos los filtros");
		System.out.println("\t5. Numérico entre tal-cual para decidir que filtros utilizar");
	}
}

