package sad.evaluation;

import java.util.Random;

import weka.attributeSelection.BestFirst;
import weka.attributeSelection.CfsSubsetEval;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.supervised.attribute.AttributeSelection;
import weka.filters.supervised.instance.Resample;
import weka.filters.unsupervised.attribute.InterquartileRange;
import weka.filters.unsupervised.attribute.Normalize;
import weka.filters.unsupervised.attribute.Remove;
import weka.filters.unsupervised.instance.RemovePercentage;
import weka.filters.unsupervised.instance.RemoveWithValues;

public class Preprocess 
{
	
	private static Preprocess miPreprocess = null;

	private Preprocess()
	{}
	
	public static Preprocess getMiPreprocess()
	{
		if (miPreprocess == null)
			miPreprocess = new Preprocess();
		return miPreprocess;
	}

	
	/**
	 * pre:se utilizan tanto el filtro como las instancias recibidas en la constructora
	 * pos:devuelve la estructura con los atributos filtrados con el filtro de tipo supervisado AtributteSelection
	 * @return Instances
	 * @throws Exception
	 */
	
	/**
	 * pre: recibe las instancias a las que aplicara el filtro
	 * @param pData
	 * @return Instances
	 * @throws Exception
	 * 
	 */
	public   Instances getFilterInstancesWithAttSelect(Instances pData) throws Exception 
	{
		CfsSubsetEval eval = new CfsSubsetEval();
		BestFirst search=new BestFirst();
		AttributeSelection fAttS= new AttributeSelection();
		fAttS.setEvaluator(eval);
		fAttS.setSearch(search);
		fAttS.setInputFormat(pData);
		// 2.1 Get new data set with the attribute sub-set
		Instances newData = Filter.useFilter(pData, fAttS);
		return newData;
	}
	
	/**
	 * pre: recibe las instancias a normalizar y el rango de normalizacion
	 * @param pData
	 * @param pRange
	 * @return Instances
	 * @throws Exception
	 * post: devuelve las instancias normalizadas
	 */
	public Instances getFilterInstancesWithNormalize(Instances pData,double pRange) throws Exception
	{
		Normalize norm = new Normalize();
		norm.setInputFormat(pData);
		norm.setScale(pRange);
		Instances normalizeData = Filter.useFilter(pData, norm);
		return normalizeData;
	}
	
	/**
	 * 
	 * @param pData
	 * @return
	 * @throws Exception
	 * 
	 * pre:Se utilizan los datos pasados por parámetro
	 * Se indican las intancias con outliers y/o extreme values
	 * y se eliminan las mismas. Tambien se elminian los atributos
	 * indicadores de outliers y extreme values.
	 * post:Se devuelven los datos sin outliers y extreme values
	 */
	public Instances getFilterInstancesWithoutOutliers(Instances pData) throws Exception{
		InterquartileRange inter= new InterquartileRange();
		//Añadimos los atributos outlier y extreme value al final de cada instancia
		inter.setOptions(weka.core.Utils.splitOptions("-R first-last -O 3.0 -E 63.0"));
		inter.setInputFormat(pData);
		Instances forFilterDataAltered = Filter.useFilter(pData, inter);
		RemoveWithValues rm = new RemoveWithValues();
		Instances data1 = null;
		Instances data2 =  null;
		//Guardamos en numero total de atributos de los datos(Los outliers y extreme
		//values se almacenan en las ultimas posiciones)
		int attributeNumber = forFilterDataAltered.numAttributes();
		rm.setInputFormat(forFilterDataAltered);
		//Trabaja con indices como en el GUI, empieza en 1, no en 0
		rm.setOptions(weka.core.Utils.splitOptions("-S 0.0 -C "+(attributeNumber)+" -L 2"));
		data1 = Filter.useFilter(forFilterDataAltered, rm);
		rm.setOptions(weka.core.Utils.splitOptions("-S 0.0 -C "+(attributeNumber - 1)+" -L 2"));
		data2 = Filter.useFilter(data1, rm);
		Remove rmAtt = new Remove();
		rmAtt.setInputFormat(data2);
		rmAtt.setOptions(weka.core.Utils.splitOptions("-R "+(data2.numAttributes()-1)+","+data2.numAttributes()));
		Instances newData = Filter.useFilter(data2, rmAtt);
		return newData;
	}
	
	/**
	 * pre: recibe las instancias desbalanceadas(más de una clase que de las demas)
	 * @param pData
	 * @return Instances
	 * @throws Exception
	 * post: devuelve las instancias balanceadas (cantidades similare de cada clase)
	 */
	public Instances getBalancedInstances(Instances pData) throws Exception{
		Resample resam = new Resample();
		resam.setOptions(weka.core.Utils.splitOptions("-B 1.0 -S 1 -Z 100.0"));
		resam.setInputFormat(pData);
		return Filter.useFilter(pData, resam);
	}
	
	/**
	 * pre: recibe un conjunto de instacias
	 * @param allData
	 * @param trainData
	 * @param testData
	 * @return  Instances[] instances[0]=trainData instances[1]=testData
	 * @throws Exception
	 * post: devuelve un array de instancias, la primera posicion contiene 
	 * las insntacias de entrenamiento (70%) y la segunda posicion las 
	 * instancias de test (30%)
	 */
	
	public Instances[] doPartition(Instances allData) throws Exception
	{
			Instances data = null;		
			data = this.randomize(allData, 42);
			RemovePercentage remover = new RemovePercentage();
			remover.setInputFormat(data);
			//Establecemos que queremos dejar el 70% de los datos;
			remover.setPercentage(30.0);
		
			Instances trainData = Filter.useFilter(data, remover);
			//Si no hacemos esto quita los el 70% de los datos que tenia
			//y nos devuelve todos los datos de alldata
			remover.setInputFormat(data);
			
			//Establecemos que queremos dejar el 30% de los datos;
			remover.setPercentage(70.0);
			
			Instances testData = Filter.useFilter(data, remover);	
			
			Instances[] trainTest = {trainData,testData};
			//En la primera posicion del array SIEMPRE estara el train, y en la segunda el test
			
			return trainTest ;
			//He comprobado que por lo menos las 2 primera instancias son distintas
	}
	/**
	 * 
	 * @param data
	 * @param seed
	 * @return Instances
	 * @throws Exception
	 */
	public Instances randomize (Instances data, int seed) throws Exception
	{
		 Random rand = new Random(seed);   // create seeded number generator
		 Instances randData = new Instances(data);   // create copy of original data
		 randData.randomize(rand);		 
		 return randData;
	}
	/**
	 * 
	 * @param pData
	 * @return Instances
	 * @throws Exception
	 */
	public Instances getFilterInstancesWithoutClass(Instances pData) throws Exception
	{
		Instances data = randomize(pData, 42);
		for (int i = 0; i < data.numInstances(); i++)
		{
			data.instance(i).setClassMissing();
		}
		return data;
	}
}
