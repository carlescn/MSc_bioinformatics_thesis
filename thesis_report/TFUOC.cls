%------------- PLANTILLA TFM UOC EIMT--------------
%--------------- v3.1 --------------------------
%--------------- Antoni Pérez-Navarro --------------------------
%--------------- Modificat per Carles Criado Ninà --------------
\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{TFUOC}[2021/09/21 LaTeX Package for UOC final projects]
\RequirePackage[utf8]{inputenc} %utf8x breaks package hyperref
\RequirePackage{ifthen}
\RequirePackage[dvips]{graphicx}
\RequirePackage{url}
\RequirePackage{fancyhdr}
%\RequirePackage{color} %Not necessary if xcolor is loaded
\RequirePackage{array}
\RequirePackage{colortbl}
%\RequirePackage{hyperref}
\RequirePackage[usenames,dvipsnames]{xcolor}
\RequirePackage{array, multirow, multicol}
\RequirePackage{stackengine}

%Set graphics path
\graphicspath{{tfuoc_img/}{img/}{figures/}}

%% ----------------------- Opcions d'entrada pròpies del paquet: català, castellà o anglès-----
\newboolean{langspanish}
\newboolean{langcatalan}
\newboolean{langenglish}
\newboolean{practica}
\newboolean{masterbib}
\DeclareOption{CAT}{
\setboolean{langcatalan}{true}
\setboolean{langspanish}{false}
\setboolean{langenglish}{false}
}
\DeclareOption{IB}{
\setboolean{langcatalan}{false}
\setboolean{langspanish}{true}
\setboolean{langenglish}{false}
}
\DeclareOption{ENG}{
\setboolean{langcatalan}{false}
\setboolean{langspanish}{false}
\setboolean{langenglish}{true}
}
\DeclareOption{BIB}{
\setboolean{masterbib}{true}
}


\DeclareOption*{\PassOptionsToClass{\CurrentOption}{report}}% Passa les opcions no declarades explícitament a "article". Així es poden fer servir totes les opcions de report.
\ProcessOptions\relax


%% --------------- Basat en l'estil article -------------------

\LoadClass[12pt]{report}
\RequirePackage{geometry}

\RequirePackage{sectsty}
\chapterfont{\color{darkblueUOC}}  % sets colour of chapters
\sectionfont{\color{darkblueUOC}}
\subsectionfont{\color{darkblueUOC}}
\subsubsectionfont{\color{darkblueUOC}}

\ifthenelse{\boolean{langspanish}}
{\RequirePackage[spanish]{babel}}%%% CASTELLÀ
	{\ifthenelse{\boolean{langcatalan}}
	{\RequirePackage[catalan]{babel}}%%% CATALÀ
	{\RequirePackage[english]{babel}}}%%% ANGLÈS

%% --------------- Configuració de la pàgina -------------------
\setlength{\topmargin}{0cm}
\setlength{\textheight}{23cm}
\setlength{\textwidth}{17cm}
\setlength{\oddsidemargin}{0.75cm}
\setlength{\evensidemargin}{0.75cm}
\setlength{\headheight}{1cm}
\setlength{\hoffset}{-1truecm}
\setlength{\voffset}{-2truecm}

%% -------------- Paquets d'idioma -------------------------




%% --------------- Capçalera de les planes d'inici de capítol
% aqui definimos el encabezado y pie de pagina de la pagina inicial de un capitulo.
\fancypagestyle{plain}{
%\fancyhead[L]{}%\scriptsize{\@author}}
%\fancyfoot[L]{}
%\fancyhead[C]{}
%\fancyhead[R]{\scriptsize \nouppercase{\leftmark}} % AQUEST ÉS EL QUE FUNCIONA
%\fancyhead[R]{\scriptsize \uppercase{\@titcrt}} % AQUEST ÉS EL QUE FUNCIONA
\fancyfoot[L]{\color{darkblueUOC}{\scriptsize{\@author}}}
\fancyfoot[C]{\color{darkblueUOC}{\thepage}}
\fancyfoot[R]{\raisebox{-.45\height}{\includegraphics[height=0.35truecm,keepaspectratio=true]{eimt.png}} \hspace*{0.25truecm} }
%\renewcommand{\headrulewidth}{0pt}
\renewcommand\footrulewidth{1mm}
\renewcommand{\footrule}{\hbox to\headwidth{%
  \color{lightblueUOC}\leaders\hrule height \footrulewidth\hfill}}
%\renewcommand{\footrule}{\hbox to\headwidth{%
%  \color{airblueuoc}\leaders\hrule height \footrulewidth\hfill}}
}

%% --------------- Capçalera de les planes normals -------------------

\pagestyle{fancy}
%\fancyhf{}
\setlength\headheight{35pt}
%\lhead{\raisebox{0.075truecm}{\includegraphics[height=1truecm]{noulogo.png}}}
\fancyfoot{}
\renewcommand\headrulewidth{0pt}
\renewcommand\footrulewidth{1mm}
\renewcommand{\footrule}{\hbox to\headwidth{%
  \color{lightblueUOC}\leaders\hrule height \footrulewidth\hfill}}
\lfoot{\includegraphics[height=0.35truecm,keepaspectratio=true]{noulogo.png}}
\lfoot{\scriptsize{\color{darkblueUOC}{\@author}}}
\cfoot{\thepage}
\rfoot{\raisebox{-.45\height}{\includegraphics[height=0.35truecm,keepaspectratio=true]{eimt.png}} \hspace*{0.25truecm} }
\fancyhead[L]
{\raisebox{-.45\height}{\includegraphics[height=1.5truecm,keepaspectratio]{noulogo.png}}}%\scriptsize{\@author}}
\fancyhead[C]{
\begin{tabular}{p{4.5truecm}}
\arrayrulecolor{lightblueUOC}\hline
%\multirow{2}{2truecm}{\@author}\\ \\
\rule{0mm}{1truecm}{}\\%Es posa per fixar l'amplada de la cel·la
\arrayrulecolor{lightblueUOC}\hline
\end{tabular}
%\scriptsize \nouppercase{\leftmark}
}
\setlength{\arrayrulewidth}{1mm}
\fancyhead[R]{
\begin{tabular}{p{5.5truecm}}
\arrayrulecolor{lightblueUOC}\hline
%\multirow{2}{2truecm}{\@author}\\ \\
\rule{0mm}{1truecm}{\color{darkblueUOC}{\scriptsize \uppercase{\@titcrt}}}\\
\arrayrulecolor{lightblueUOC}\hline
\end{tabular}
%\scriptsize \nouppercase{\leftmark}
}

%\fancyhf{}
%\fancyhead[LE]{\scriptsize \@author}
%\fancyhead[RE]{\scriptsize \@titcrt}
%\fancyhead[RO]{\scriptsize \@author}
%%\fancyhead[RO]{\scriptsize \nouppercase{\leftmark}}
%\fancyhead[LO]{\scriptsize \nouppercase{\leftmark}}
%\renewcommand{\chaptermark}[1]{ \markboth{\chaptername\ \thechapter.\ #1}{} }
%\fancyfoot[LO]{\thepage \hspace*{0.25truecm} \includegraphics[height=0.35truecm,keepaspectratio=true]{eimt.png} }
%\fancyfoot[C]{}
%\fancyfoot[RE]{\includegraphics[height=0.35truecm,keepaspectratio=true]{eimt.png} \hspace*{0.25truecm} \thepage}
%% --------------- Definició de comandes -------------------
%\newcommand{\bluepac}{\color{airblueuoc}} %% Definició del color blau de la plantilla.
\definecolor{airblueuoc}{rgb}{0.36, 0.54, 0.66}
\definecolor{lightblueUOC}{RGB}{147,234,252}
\definecolor{darkblueUOC}{RGB}{1,0,115}
\definecolor{greyUOC}{RGB}{239,239,239}
\newcommand{\bluepac}{\color{airblueuoc}} %% Definició del color blau de la plantilla.
\newcommand{\graypac}{\color{gray}} %% Definició del color gris de la plantilla.
\newcommand{\sectionblue}[1]{\section*{\bluepac #1}}
\newcommand{\subsectiongray}[1]{\section*{\graypac #1}}
\newcommand{\HRule}{\noindent\rule{\linewidth}{1.5pt}}
\newcommand{\Hrule}{\noindent\rule{\linewidth}{0.75pt}}
%{\color{airblueuoc}\hrule}
%\newcommand{\headrulecolor}[4]{\patchcmd{\headrule}{\hrule}{\color{#1}\hrule}{}{}}
%\renewcommand{\headrulewidth}{2pt}% 2pt header rule
%\renewcommand{\headrule}{\hbox to\headwidth{\color{headercolor}\leaders\hrule height \headrulewidth\hfill}}


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% ----------------------- Tria de llicència -----------------------------
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\newcommand{\@licenc}{}
\newcommand{\licenc}[1]{
\renewcommand{\@licenc}{#1}
}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% ----------------------- Generació de les metadades del TFM -----------------------------
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Explicat a: https://www.dickimaw-books.com/latex/admin/html/clsform.shtml

\newcommand{\@titcrt}{}
\newcommand{\titcrt}[1]{
\renewcommand{\@titcrt}{#1}
}

\newcommand{\@nomPDC}{}

\newcommand{\nomPDC}[1]{
\renewcommand{\@nomPDC}{#1}
}

\newcommand{\@nomPRA}{}

\newcommand{\nomPRA}[1]{
\renewcommand{\@nomPRA}{#1}
}
\newcommand{\@titulac}{}

\newcommand{\titulac}[1]{
\renewcommand{\@titulac}{#1}
}


\newcommand{\@area}{}

\newcommand{\area}[1]{
\renewcommand{\@area}{#1}
}

\newcommand{\@idioma}{}

\newcommand{\idioma}[1]{
\renewcommand{\@idioma}{#1}
}

\newcommand{\@credits}{}

\newcommand{\credits}[1]{
\renewcommand{\@credits}{#1}
}

\newcommand{\@parcla}{}

\newcommand{\parcla}[1]{
\renewcommand{\@parcla}{#1}
}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% ----------------------- Definició del resum -----------------------------
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\newcommand{\@abstractidioma}{}

\newcommand{\abstractidioma}[1]{
\renewcommand{\@abstractidioma}{#1}
}

\newcommand{\@abstractenglish}{}

\newcommand{\abstractenglish}[1]{
\renewcommand{\@abstractenglish}{#1}
}

%Definició de \escriu per fer proves.

\newcommand{\escriu}{
I aquí tens el resultat \@parcla i el títol és: \@title
}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% CREACIÓ DE L'ESTRUCTURA INICIAL DEL DOCUMENT
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\newcommand{\estructura}
{

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%%% Definició dels codis d'idioma
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	\ifthenelse{\boolean{langspanish}} %%% CASTELLÀ
	{


\newcommand{\fitxacode}{Ficha Del Trabajo Final}
\newcommand{\titolcode}{Título del trabajo:}
\newcommand{\nomautorcode}{Nombre del autor/a:}
\newcommand{\nompdccode}{Nombre del tutor/a de TF:}
\newcommand{\nompracode}{Nombre del/de la PRA:}
\newcommand{\datalliuramentcode}{Fecha de entrega:}
\newcommand{\titulaciocode}{Titulación o programa:}
\newcommand{\areacode}{Área del trabajo final:}
\newcommand{\idiomacode}{Idioma del trabajo:}
\newcommand{\creditscode}{Núm. de créditos:}
\newcommand{\paraulesclaucode}{Palabras clave:}
\newcommand{\resumcode}{Resumen del trabajo}
%-----Textos llicència
\newcommand{\ccByNcNd}{
Esta obra esta sujeta a una licencia de Reconocimiento-NoComercial-SinObraDerivada \\ \url{https://creativecommons.org/licenses/by-nc/3.0/es/}}
\newcommand{\ccByNcSa}{
Esta obra esta sujeta a una licencia de Reconocimiento-NoComercial-CompartirIgual \\ \url{https://creativecommons.org/licenses/by-nc/3.0/es/}}
\newcommand{\ccByNc}{
Esta obra esta sujeta a una licencia de Reconocimiento-NoComercial  \\ \url{https://creativecommons.org/licenses/by-nc/3.0/es/}}
\newcommand{\ccByNd}{
Esta obra esta sujeta a una licencia de Reconocimiento-SinObraDerivada \\ \url{https://creativecommons.org/licenses/by-nc/3.0/es/}}
\newcommand{\ccBySa}{
Esta obra esta sujeta a una licencia de Reconocimiento-CompartirIgual \\  \url{https://creativecommons.org/licenses/by-nc/3.0/es/}}
\newcommand{\ccBy}{
Esta obra esta sujeta a una licencia de Reconocimiento
 \\  \url{https://creativecommons.org/licenses/by-nc/3.0/es/}}
 \newcommand{\GNU}{
 Copyright \copyright  \@author\\

Permission is granted to copy, distribute and/or modify this document under the terms of the GNU Free Documentation License, Version 1.3 or any later version published by the Free Programari Foundation; with no Invariant Sections, no Front-Cover Texts, and no Back-Cover Texts.
A copy of the license is included in the section entitled "GNU Free Documentation License".
}

 \newcommand{\copyrightreservado}{
\copyright   (\@author)\\
Reservados todos los derechos. Está prohibida la reproducción total o parcial de esta obra por cualquier medio o procedimiento, comprendidos la impresión, la reprografía, el microfilm, el tratamiento informático o cualquier otro sistema, así como la distribución de ejemplares mediante alquiler y préstamo, sin la autorización escrita del autor o de los límites que autorice la Ley de Propiedad Intelectual.
}

}
	{}
	\ifthenelse{\boolean{langcatalan}} %%% CATALÀ
	{


\newcommand{\fitxacode}{Fitxa del Treball Final}
\newcommand{\titolcode}{Títol del treball:}
\newcommand{\nomautorcode}{Nom de l'autor/a:}
\newcommand{\nompdccode}{Nom del Tutor/a del TF:}
\newcommand{\nompracode}{Nom del de/la PRA:}
\newcommand{\datalliuramentcode}{Data de lliurament:}
\newcommand{\titulaciocode}{Titulació o programa:}
\newcommand{\areacode}{Àrea del treball final:}
\newcommand{\idiomacode}{Idioma del treball:}
\newcommand{\creditscode}{Núm. de crèdits:}
\newcommand{\paraulesclaucode}{Paraules clau:}
\newcommand{\resumcode}{Resum del treball}

%-----Textos llicència
\newcommand{\ccByNcNd}{
Aquesta obra està subjecta a una llicència de Reconeixement-NoComercial-SenseObraDerivada \\ \url{https://creativecommons.org/licenses/by-nc/3.0/es/}}
\newcommand{\ccByNcSa}{
Aquesta obra està subjecta a una llicència de Reconeixement-NoComercial-CompartirIgual \\ \url{https://creativecommons.org/licenses/by-nc/3.0/es/}}
\newcommand{\ccByNc}{
Aquesta obra està subjecta a una llicència de Reconeixement-NoComercial  \\ \url{https://creativecommons.org/licenses/by-nc/3.0/es/}}
\newcommand{\ccByNd}{
Aquesta obra està subjecta a una llicència de Reconeixement-SenseObraDerivada \\ \url{https://creativecommons.org/licenses/by-nc/3.0/es/}}
\newcommand{\ccBySa}{
Aquesta obra està subjecta a una llicència de Reconeixement-CompartirIgual  \\ \url{https://creativecommons.org/licenses/by-nc/3.0/es/}}
\newcommand{\ccBy}{
Aquesta obra està subjecta a una llicència de Reconeixement
 \\\  \url{https://creativecommons.org/licenses/by-nc/3.0/es/}}
 \newcommand{\GNU}{
 Copyright \copyright  \@author\\
Permission is granted to copy, distribute and/or modify this document under the terms of the GNU Free Documentation License, Version 1.3 or any later version published by the Free Programari Foundation; with no Invariant Sections, no Front-Cover Texts, and no Back-Cover Texts.
A copy of the license is included in the section entitled "GNU Free Documentation License".
}
 \newcommand{\copyrightreservado}{
\copyright   (\@author)\\
Reservats tots els drets. Està prohibit la reproducció total o parcial d'aquesta obra per qualsevol mitjà o procediment, compresos la impressió, la reprografia, el microfilm, el tractament informàtic o qualsevol altre sistema, així com la distribució d'exemplars mitjançant lloguer i préstec, sense l'autorització escrita de l'autor o dels límits que autoritzi la Llei de Propietat Intel·lectual.
}
}
	{}\ifthenelse{\boolean{langenglish}}  %%% ANGLÈS
	{

\newcommand{\fitxacode}{Final Work Card}
	\newcommand{\titolcode}{Title of the work:}
\newcommand{\nomautorcode}{Name of the author:}
\newcommand{\nompdccode}{Name of the tutor:}
\newcommand{\nompracode}{Name of the SRP:} %toni: no sé si es diuen així
\newcommand{\datalliuramentcode}{Date of delivery:}
\newcommand{\titulaciocode}{Studies or Prgram:}
\newcommand{\areacode}{Area or the Final Work:}
\newcommand{\idiomacode}{Language of the work:}
\newcommand{\creditscode}{Number of credits:}
\newcommand{\paraulesclaucode}{Keywords:}
\newcommand{\resumcode}{Abstract}


%-----Textos llicència
\newcommand{\ccByNcNd}{
This work is under the license Attribution-NonCommercial-NoDerivatives \\ \url{https://creativecommons.org/licenses/by-nc/3.0}}
\newcommand{\ccByNcSa}{
This work is under the license Attribution-NonCommercial-ShareAlike \\ \url{https://creativecommons.org/licenses/by-nc/3.0}}
\newcommand{\ccByNc}{
This work is under the license Attribution-NonCommercial \\ \url{https://creativecommons.org/licenses/by-nc/3.0}}
\newcommand{\ccByNd}{
This work is under the license Attribution-NoDerivatives \\ \url{https://creativecommons.org/licenses/by-nc/3.0}}
\newcommand{\ccBySa}{
This work is under the license Attribution-ShareAlike  \\ \url{https://creativecommons.org/licenses/by-nc/3.0}}
\newcommand{\ccBy}{
This license lets others distribute, remix, adapt, and build upon your work, even commercially, as long as they credit you for the original creation. This is the most accommodating of licenses offered. Recommended for maximum dissemination and use of licensed materials.
 \\\  \url{https://creativecommons.org/licenses/by-nc/3.0}}
 \newcommand{\GNU}{
 Copyright \copyright  \@author\\
Permission is granted to copy, distribute and/or modify this document under the terms of the GNU Free Documentation License, Version 1.3 or any later version published by the Free Programari Foundation; with no Invariant Sections, no Front-Cover Texts, and no Back-Cover Texts.
A copy of the license is included in the section entitled "GNU Free Documentation License".
}
 \newcommand{\copyrightreservado}{
\copyright   (\@author)\\
All rights reserved. Reproduction in whole or in part of this work by any means or procedure, including printing, reprographics, microfilm, computer processing or any other system, as well as the distribution of copies by rental and loan, without the written authorization from the author or the limits authorized by the Intellectual Property Law.
}
}{}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%---------------------------- Plana del títol
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\newcommand{\planatitol}
{
\begin{titlepage}
\thispagestyle{empty}
\newgeometry{left=20mm, top=30mm,textheight=26cm,textwidth=17cm}
\begin{tabular}[t]{p{6.25cm}p{0.25cm}p{10.5cm}}
\multicolumn{3}{c}{
\cellcolor{lightblueUOC}{
\begin{minipage}[t][7cm][t]{\textwidth}
\vspace*{\fill}
\Huge{\textbf{\@title}}
\vspace*{\fill}
\end{minipage}}}\\
&&\\
\ifthenelse{\boolean{masterbib}}
{\begin{minipage}{\textwidth}
\includegraphics[width=6.25cm]{UOCllargub.jpg}
\end{minipage}
}
{\begin{minipage}{\textwidth}
\includegraphics[width=6.25cm]{UOCllarg.jpg}
\end{minipage}
}

%
&&
\cellcolor{greyUOC}{
\begin{minipage}{\textwidth}
\Large{\textcolor{darkblueUOC}{\textbf{\@author}}}

\vspace*{1cm}
\textcolor{darkblueUOC}{\@area}

\vspace*{1cm}

\textcolor{darkblueUOC}{\@titulac}

\vspace*{1cm}

\textcolor{darkblueUOC}{\nompdccode}\\
\textcolor{darkblueUOC}{\textbf{\@nomPDC}}

\vspace*{1cm}
\textcolor{darkblueUOC}{\nompracode}\\
\textcolor{darkblueUOC}{\@nomPRA}

\vspace*{1cm}

\textcolor{darkblueUOC}{\@date}
\vspace*{\fill}
\end{minipage}
}
\end{tabular}
\restoregeometry
\end{titlepage}
}



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%---------------------------- Llicència
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\ifthenelse{\equal{\@licenc}{ccByNcNd}}
  {% True case

   \newcommand{\llicencia}{\includegraphics[scale=1]{./licenc/ccbyncnd.png}\\
   \ccByNcNd}%
  }
  {
	  \ifthenelse{\equal{\@licenc}{ccByNcSa}}
	  {% True case

	   \newcommand{\llicencia}{\includegraphics[scale=1]{./licenc/ccbyncsa.png}\\
	   \ccByNcSa}%
	  }
	  {
	  \ifthenelse{\equal{\@licenc}{ccByNc}}
	  {% True case
		   \newcommand{\llicencia}{\includegraphics[scale=1]{./licenc/ccbync.png}\\
		   \ccByNc}%
		  }
		  {
			  \ifthenelse{\equal{\@licenc}{ccByNd}}
			  {% True case

			   \newcommand{\llicencia}{\includegraphics[scale=1]{./licenc/ccbynd.png}\\
			   \ccByNd}%
			  }
			  {
				  \ifthenelse{\equal{\@licenc}{ccBySa}}
				  {% True case

				   \newcommand{\llicencia}{\includegraphics[scale=1]{./licenc/ccbysa.png}\\
				   \ccBySa}%
				  }
				  {
					   \ifthenelse{\equal{\@licenc}{ccBy}}
					  {% True case

					   \newcommand{\llicencia}{\includegraphics[scale=1]{./licenc/ccby.png}\\
					   \ccBy}%
					  }
					  {
					    \ifthenelse{\equal{\@licenc}{GNU}}
					  {% True case
					   \newcommand{\llicencia}{\GNU}%
					  }
					  {
					      \ifthenelse{\equal{\@licenc}{copyright}}
					  {% True case

					   \newcommand{\llicencia}{\copyrightreservado}%
					  }
					  {
					  \newcommand{\llicencia}{NO LICENSE}
					  }
}}}}}}}


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%---------------------------- Construcció de l'estructura
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\planatitol

\newpage
\llicencia
\newpage

\vspace{\fill}
\section*{\fitxacode}
%%	\HRule
%% Ayuda con la tabla: http://metodos.fam.cie.uva.es/~latex/apuntes/apuntes10.pdf
\begin{center}
\begin{tabular}{|>{\bfseries}r|>{\columncolor[gray]{0.9}}p{10truecm}|}
%\begin{tabular}{|r|p{10truecm}|}
\hline
\textcolor{darkblueUOC}{\titolcode} & \@title\\
\hline
\textcolor{darkblueUOC}{\nomautorcode} & \@author\\
\hline
\textcolor{darkblueUOC}{\nompdccode} &\@nomPDC\\
\hline
\textcolor{darkblueUOC}{\nompracode} & \@nomPRA \\
\hline
\textcolor{darkblueUOC}{\datalliuramentcode} & \@date\\
\hline
\textcolor{darkblueUOC}{\titulaciocode} & \@titulac\\
\hline
\textcolor{darkblueUOC}{\areacode} & \@area\\
\hline
\textcolor{darkblueUOC}{\idiomacode} & \@idioma\\
\hline
%\creditscode & \@credits\\
%\hline
\textcolor{darkblueUOC}{\paraulesclaucode} & \@parcla\\
\hline
\end{tabular}

\end{center}
\vspace{\fill}

\newpage

\ifthenelse{\boolean{langenglish}}  %%% ANGLÈS
{
	\begin{tabular}{|p{15truecm}|}
	\hline
	\rowcolor[gray]{0.9}\textcolor{darkblueUOC}{\textbf{Abstract}}\\
	\hline
	\@abstractenglish\\
	\hline
	\end{tabular}
}{
	\begin{tabular}{|p{15truecm}|}
	\hline
	\rowcolor[gray]{0.9}\textcolor{darkblueUOC}{\textbf{\resumcode}}\\
	\hline
	\@abstractidioma\\
	\hline
	\end{tabular}

	\newpage

	\begin{tabular}{|p{15truecm}|}
	\hline
	\rowcolor[gray]{0.9}\textbf{Abstract}\\
	\hline
	\@abstractenglish\\
	\hline
	\end{tabular}
	}
}
\pagestyle{fancy}

%%% https://es.overleaf.com/learn/latex/How_to_Write_a_Thesis_in_LaTeX_(Part_5)%3A_Customising_Your_Title_Page_and_Abstract


\relax

