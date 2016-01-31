package sad.evaluation;

import java.util.Random;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.functions.LibSVM;
import weka.core.Instances;

@SuppressWarnings("serial")
public class Multibounds extends Evaluation
{
	private Instances data;
	
	
	public Multibounds(Instances pData) throws Exception
	{
		super(pData);
		this.data = pData;
	}
	

	private Instances getData() {
		return data;
	}


	private void setData(Instances pData) 
	{
		this.data = pData;
	}
	
	


	/**
	 * @param newData
	 * @param estimador
	 * @param pFold
	 * @return evaluator
	 * 
	 * pre:Recibe el conjunto de instancias con las que se evalúa, además el clasificador
	 * debe estar previamente configurado y recibido como parametro en la constructora
	 * de la presente clase.
	 * 
	 * pos:Devuelve un Evaluator del clasificador recibido en la constructora.
	 * Evalua desempeño del clasificador utilizando el método pFold-HCV
	 * @throws Exception 
	 */
	public  void assesPerformanceNFCV(Classifier estimador, int pFold, Instances pData) throws Exception
		{
		try 
		{			
			estimador.buildClassifier(pData);
		} catch (Exception e2) 
		{
			// TODO Auto-generated catch block
			e2.printStackTrace();
		}
		this.crossValidateModel(estimador, pData,pFold, new Random(1));
		
		
		// Random(1): the seed=1 means "no shuffle" :-!
		/*Alternativa al método toMatrixString()
		 * double confMatrix[][]= evaluator.confusionMatrix();
		for(int row=0; row<confMatrix.length;row=row++){
			for(int col=0; col<confMatrix.length;col++){
				System.out.print(confMatrix[row][col]);
				System.out.print("|");
			}
			System.out.println();
		}*/
		
		/*System.out.println(evaluator.toSummaryString("\nResults\n======\n",true));//imprime los datos estadisticos
		System.out.println(evaluator.toClassDetailsString());
		System.out.println(evaluator.toMatrixString());//imprime la matriz de confusion
		System.out.println("weightedFMeasure: "+evaluator.weightedFMeasure());*/
		
		}
	/**
	 * TODO pre: Recibe el evaluador con la evaluacion hecha
	 * fuente:http://weka.wikispaces.com/Programmatic+Use TODO
	 * @param evaluator
	 * @throws Exception
	 *  post: imprime por pantalla los resultados de la evaluacion
	 */
	public void printStatistics(Evaluation evaluator)
	{
		try {
			System.out.println(evaluator.toClassDetailsString());
			System.out.println(evaluator.toMatrixString());
			System.out.println("weightedFMeasure: "+evaluator.weightedFMeasure());
		} catch (Exception e) {
			e.printStackTrace();
		}		
	}
	
	/**
	 * pre : recibe el modelo ya entrenado
	 * @palo  estimador
	 * @param unlabeled
	 * @return Instances
	 * @throws Exception
	 *	post: devuelve el conjunto de instancias con las predicciones hechas
	 */
	public Instances predictionsMaker(LibSVM pEstimador,
			Instances pUnlabeled) throws Exception {
		// Se crea una copia de las instancias sin clasificar para obtener después la clase estimada
		 Instances labeled = new Instances(pUnlabeled);
		 pEstimador.setDoNotReplaceMissingValues(false);
		 pEstimador.setProbabilityEstimates(true);
		 // Estimar las clases
		 for (int i = 0; i < pUnlabeled.numInstances(); i++) {
		   double clsLabel = pEstimador.classifyInstance(pUnlabeled.instance(i));
		   //double[] clsDis = pEstimador.distributionForInstance(pUnlabeled.instance(i)); //para obtener la distribución de probabilidad
		   labeled.instance(i).setClassValue(clsLabel);
		   
		 }
		return labeled;
	}
	/**
	 * TODO pre-post
	 * @param pEstimador
	 * @param pUnlabeled
	 * @return Instances
	 * @throws Exception
	 */
	public  Instances predictionsMakerGeneric(Classifier pEstimador,
			Instances pUnlabeled) throws Exception {
		// Se crea una copia de las instancias sin clasificar para obtener después la clase estimada
		 Instances labeled = new Instances(pUnlabeled);
		 
		 // Estimar las clases
		 for (int i = 0; i < pUnlabeled.numInstances(); i++) {
		   double clsLabel = pEstimador.classifyInstance(pUnlabeled.instance(i));
		   labeled.instance(i).setClassValue(clsLabel);
		 }
		return labeled;
	}
	/**
	 * TODO pre-post
	 * fuente:http://weka.wikispaces.com/Programmatic+Use TODO
	 * @param estimador
	 * @param pData
	 * @param pRuta
	 * @throws Exception
	 */
	public void dishonestEvaluator(Classifier estimador, Instances pData) throws Exception
	{		
		estimador.buildClassifier(pData);		
		this.evaluateModel(estimador, pData);	 
	}
	
	/**
	 * TODO pre-post
	 * fuente:http://weka.wikispaces.com/Programmatic+Use TODO
	 * @param estimador
	 * @param pData
	 * @param pRuta
	 * @throws Exception
	 */
	public void dishonestEvaluatorSVM(LibSVM estimador, Instances pData) throws Exception
	{
		
		estimador.buildClassifier(pData);		
		this.evaluateModel(estimador, pData);
		 
	}
	
}
