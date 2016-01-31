package sad.evaluation;

import sad.utils.VerboseCutter;
import weka.classifiers.Evaluation;
import weka.classifiers.functions.LibSVM;
import weka.classifiers.meta.CVParameterSelection;
import weka.core.Instances;
import weka.core.SelectedTag;

@SuppressWarnings("serial")
public class GridSearchWithCVParam extends Evaluation
{

	private static GridSearchWithCVParam miGSearch;
		
		
	private  GridSearchWithCVParam(Instances pData) throws Exception
	{
		super(pData);	
	}
		
	public static GridSearchWithCVParam getGSParams(Instances pData) throws Exception
	{
		if (miGSearch == null)
			miGSearch = new GridSearchWithCVParam(pData);
		return miGSearch;
	}
	public void getBestParamsSVM(Instances pData,boolean foldC) throws Exception
	{	
		System.out.println("------Starting parameter optimization with CVParameterSelection------");
		System.out.println("This might take a while...");
		VerboseCutter.getVerboseCutter().cutVerbose();

		CVParameterSelection cv = new CVParameterSelection();
		CVParameterSelection cv2 = new CVParameterSelection();
		LibSVM svm = new LibSVM();
		LibSVM svm2 = new LibSVM();
		
		svm.setSVMType(new SelectedTag(0, LibSVM.TAGS_SVMTYPE));
		svm2.setSVMType(new SelectedTag(0, LibSVM.TAGS_SVMTYPE));
		
		svm.setKernelType(new SelectedTag(2,LibSVM.TAGS_KERNELTYPE));
		svm2.setKernelType(new SelectedTag(2,LibSVM.TAGS_KERNELTYPE));
		cv.setClassifier(svm);
		cv2.setClassifier(svm2);
		
		cv.addCVParameter("C 0.000030517578125 8.0 16.0");
		cv.addCVParameter("G 0.125 16.0 8.0");
		cv2.addCVParameter("C 8.0 1024.0 16.0");
		cv2.addCVParameter("G 0.125 16.0 8.0");
		
		cv.buildClassifier(pData);
		cv2.buildClassifier(pData);		
		
		
		Multibounds evaluator1 = new Multibounds(pData);
		Multibounds evaluator2 = new Multibounds(pData);
		Multibounds evaluatorBest = new Multibounds(pData);
		
		if (foldC)
		{
			evaluator1.assesPerformanceNFCV(cv, 5,pData);
			evaluator2.assesPerformanceNFCV(cv, 5, pData);
		}
		else
		{
			evaluator1.dishonestEvaluator(cv, pData);
			evaluator2.dishonestEvaluator(cv, pData);
		}
		String bestC, bestG;
		
		if(evaluator1.rootMeanSquaredError()<evaluator2.rootMeanSquaredError())
		{
			evaluatorBest=evaluator1;
			bestC = cv.getBestClassifierOptions()[1];
			bestG = cv.getBestClassifierOptions()[3];
		}
		else
		{
			evaluatorBest=evaluator2;
			bestC = cv2.getBestClassifierOptions()[1];
			bestG = cv2.getBestClassifierOptions()[3];
		}		
		VerboseCutter.getVerboseCutter().activateVerbose();
		
		System.out.println("----------GridSearchSVM---------------------");
		System.out.println("Mejor C:\t"
			+ bestC);
		System.out.println("Mejor gamma:\t"
				+ bestG);
		System.out.println(evaluatorBest.toSummaryString()+"Recall:\t " + evaluatorBest.weightedRecall() + "\nPrecision:\t " + 
						evaluatorBest.weightedPrecision() + "\n" + evaluatorBest.toMatrixString()
						+"\n" +"weightedROC : " + evaluatorBest.weightedAreaUnderROC() + "\n con F-measure:" + 
						evaluatorBest.weightedFMeasure());
		
		

	}
}
