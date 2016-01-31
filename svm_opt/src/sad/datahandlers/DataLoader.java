package sad.datahandlers;

import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.text.SimpleDateFormat;
import java.util.Calendar;
import java.util.GregorianCalendar;
import java.util.Random;
import java.io.BufferedWriter;

import weka.core.Instances;
import weka.core.converters.ArffSaver;

//TOTO otros formatos

public class DataLoader 

{
	private FileReader FR;
	private String path;
	static final int classPos=1;
	
	/**
	 * pre:Se recibe como parámetro
	 * La constructora comprueba que el path que le pasamos como parametro es correcto antes de 
	 * construir correcto, tratando la exce
	 * @param path
	 * @param data
	 */
	public DataLoader(String pPath) 
	{
		
		try 
		{
			this.path=pPath;
			this.FR= new FileReader(path); 
		} 
		catch (FileNotFoundException e)
		{
			System.out.println("ERROR: Revisar path del fichero de datos:"+path);
		}
		
	}
	
	private FileReader getFR() 
	{
		return FR;
	}
	@SuppressWarnings("unused")
	private void setFR(FileReader fR) 
	{
		FR = fR;
	}
	
	
	
	/**
	 * pre:
	 * pos:carga como instancias el fichero instanciado previamente en la constructora de la clase.
	 * 
	 */
	public   Instances instancesLoader()
	{	
		Instances data=null;
		try 
		{
			data = new Instances(this.getFR());
		} catch (IOException e) {
			System.out.println("ERROR: Revisar contenido del fichero de datos: "+this.path);
		}
		// Close the file
		this.closeFR();
		// Se aplicará siempre que se carguen las instancias.
		//this.instanceShuffle(data);
		this.selectClass(data);
		return data;
	}
	
	/**
	 * Close the file
	 */
	private void closeFR(){
		try {
			this.getFR().close();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
	
	/**
	 * pre:recibe como parámetro las instancias a barajar pos:las mezcla con el filtro randomize
	 */
	@SuppressWarnings("unused")
	private void instanceShuffle(Instances pData)
	{
		Random random = new Random();
		pData.randomize(random);
	}
	/**
	 * Specify which attribute will be used as the class: the last one, in this case 
	 */
	private void selectClass(Instances pData){
		pData.setClassIndex(pData.numAttributes()-classPos);
	}
	

	/**
	 * pre:el directorio fichero_datos debe existir en el directorio del proyecto?(revisar)
	 * @param test
	 * @throws IOException
	 * @throws FileNotFoundException
	 */
	public  void pushDataSets(Instances pDataSet,String pNomFile,String pPath)  
		{
			Calendar calendar = new GregorianCalendar(); // Fecha y hora actuales.
			SimpleDateFormat dateFormat = new SimpleDateFormat("yyyyMMdd-HHmm"); // Formato de la fecha.
			String dateS = dateFormat.format(calendar.getTime()); // Fecha y hora actuales formateadas.
			
			// Variable para captura ranking y variable para guardar el contenido en un archivo.
			String path = pPath.substring(0, pPath.length() - 5) + pNomFile + dateS + ".arff";
			
			ArffSaver guardarARFF = new ArffSaver();        
			guardarARFF.setInstances(pDataSet);
			try {
				guardarARFF.setDestination(new FileOutputStream(path));
			} catch (FileNotFoundException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			} catch (IOException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
			try {
				guardarARFF.writeBatch();
			} catch (IOException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}//Escribe el lote de instancias
		}
	/**
	 * pre:
	 * @param FilePath = String, indica la ruta del archivo.
	 * @param FileContent = String, indica el contenido del archivo
	 * @param CleanFileContent = boolean.Si tue y existe el archivo borra el contenido, 
	 * 			si false, añade el contenido al final del archivo.
	 * @return true si se guarda con éxito, false en caso contrario.
	 * 
	 * fuente : http://www.creatusoftware.com/index.php?option=com_content&view=article&id=142:funcion-para-guardar-un-archivo-en-java&catid=62:fuentes-java&Itemid=41
	 */
	
	public boolean SaveFile(String FilePath, String FileContent, boolean CleanFileContent)
	{
	 
	    FileWriter file;
	    BufferedWriter writer;
	     
	    try
	    {
	        file = new FileWriter(FilePath, !CleanFileContent);
	        writer = new BufferedWriter(file);
	        writer.write(FileContent, 0, FileContent.length());
	         
	        writer.close();
	        file.close();
	 
	        return true;
	    } 
	    catch (IOException ex) 
	    {
	        ex.printStackTrace();
	        return false;
	   }
	}

}
