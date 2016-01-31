package sad.evaluation;

import sad.utils.VerboseCutter;
import weka.classifiers.Evaluation;
import weka.classifiers.meta.CVParameterSelection;
import weka.classifiers.rules.OneR;
import weka.core.Instances;

@SuppressWarnings("serial")
public class ScanCVParams extends Evaluation
{
	private static ScanCVParams myScanCVParams = null;
	
	private ScanCVParams (Instances pData) throws Exception
	{super(pData);	}
	
	public static ScanCVParams getScanCVParams(Instances pData) throws Exception
	{
		if (myScanCVParams == null)
			myScanCVParams = new ScanCVParams(pData);
		return myScanCVParams;
	}
	/**
	 * pre:Los datos se deben preprocesar previamente.
	 * post:muestra por consola la optimización del Baseline OneR y las figuras de mérito.
	 * @param pData
	 * @param foldC
	 * @throws Exception
	 */
	public void getBestParamsOneR(Instances pData, boolean foldC) throws Exception
	{
		CVParameterSelection cvParameterSelection = new CVParameterSelection();
		cvParameterSelection.setClassifier(new OneR());
		cvParameterSelection.addCVParameter("B 1.0 " + pData.numInstances()+" " + pData.numInstances());
		Multibounds evaluator = new Multibounds(pData);
		
		VerboseCutter.getVerboseCutter().cutVerbose();
		if (foldC)
			evaluator.assesPerformanceNFCV(cvParameterSelection, 5,pData);
		else
			evaluator.dishonestEvaluator(cvParameterSelection, pData);
		VerboseCutter.getVerboseCutter().activateVerbose();
		
		System.out.println("----------CVBaseline---------------------");
		System.out.println("Mejor " + cvParameterSelection.getBestClassifierOptions()[0] + ":\t"
			+ cvParameterSelection.getBestClassifierOptions()[1]);
		System.out.println(evaluator.toSummaryString()+"Recall:\t " + evaluator.weightedRecall() + "\nPrecision:\t " + 
						evaluator.weightedPrecision() + "\n" + evaluator.toMatrixString()
						+"\n" +"weitheredROC : " + evaluator.weightedAreaUnderROC() + "\n con F-measure:" + 
						evaluator.weightedFMeasure());
	}
	
}
