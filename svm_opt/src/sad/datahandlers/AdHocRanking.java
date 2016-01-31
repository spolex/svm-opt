package sad.datahandlers;



public class AdHocRanking {

	private int bBest;
	private String statistics;
	private double bestC;
	private double bestG;
	
	public AdHocRanking(double bestC, double bestG,
			String summary) 
	{
		this.statistics=summary;
		this.bestC=bestC;
		this.bestG=bestG;
	}
	
public AdHocRanking( int pBBest,  String pStatistics) 
	{		
		this.statistics=pStatistics;
		this.bBest=pBBest;	
	}

	public double getBestC() {
		return bestC;
	}

	public double getBestG() {
		return bestG;
	}	
	
	private int getbBest(){
		return this.bBest;
	}

	private String getStatistics() {
		return statistics;
	}
	/**
	 * pre : recibe un Double (Solo para OneR)
	 * @palo  estimador
	 * @param unlabeled
	 * @return String
	 * post : El String contiene el mensaje para imprimir
	 */
	public String toStringOneR(double pFmeasure)
	{
		String ans=new String();
		
		
			ans=("\nEste clasificador ha obtenido una fMeasure de "+pFmeasure+
					"\nUtilizando una B de: "+this.getbBest()+"\n"+
					this.getStatistics()+"\n");
		
		return ans;
	}
	/**
	 * pre : recibe un Double (Solo par SVM)
	 * @palo  estimador
	 * @param unlabeled
	 * @return String
	 * post : El String contiene el mensaje para imprimir
	 */
	public String toStringSVM(double pFmeasure)
	{
		String ans=new String();
		
		
			ans=("\nEste clasificador ha obtenido una fMeasure de "+pFmeasure+
					"\nUtilizando una C de: "+this.getBestC()+"\n"+
					"\nUtilizando un  gamma de: "+this.getBestC()+"\n"+
					this.getStatistics()+"\n");
		
		return ans;
	}
}
