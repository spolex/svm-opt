package sad.utils;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.PrintStream;



public class VerboseCutter {

	private static VerboseCutter myVerboseCutter = null;
	
	//Para almacenas la salida estándar y la salida con error por defgecto
	private PrintStream standardOutput;
	private PrintStream errorOutput;
	
	
	//Las salidas a las que redirigiremos
	private File file;
	private PrintStream temporaryOutput;
	
	private VerboseCutter(){
		
	}
	
	public static VerboseCutter getVerboseCutter()
	{
		if (myVerboseCutter == null)
			myVerboseCutter = new VerboseCutter();
		return myVerboseCutter;
	}

	/**
	 * Desvía tanto la salida estándar como la salida con error a
	 * un fichero temporal para que no se muestren por pantalla
	 */
	public void cutVerbose()
	{
		//Guardamos out y err de system
		standardOutput = System.out;
		errorOutput = System.err;
		
		
		try {
			//Creamos un fichero con un nombre que sea dificil que se repita
			file = new File("./Bundesnachrichtendienst.xyzww");
			
			//Redirigimos al fichero de nombre dificil de repetir
			temporaryOutput = new PrintStream(file);
			
			//Redirigimos error y estándar
			System.setOut(temporaryOutput);
			System.setErr(temporaryOutput);
			
			//Borramos el fichero al cerrar
			file.deleteOnExit();		
			
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}
	}

	/**
	 * Restaura los valores por defecto de la salida estándar y
	 * la salida con error. Se volverán a mostrar por pantalla
	 */
	public void activateVerbose()
	{
		//Redirigimos al destino inicial las salidas de error y estandar
		System.setErr(errorOutput);
		System.setOut(standardOutput);
		
		//Cerramos el fichero y lo borramos
		temporaryOutput.close();
	}

}
