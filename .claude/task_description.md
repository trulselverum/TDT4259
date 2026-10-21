\documentclass[11pt,a4paper]{article}

% ---------- Packages ----------
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage[english]{babel}
\usepackage[margin=2.5cm, top=2cm]{geometry} % slightly smaller top margin
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{tabularx}
\usepackage{float}
\usepackage{caption}
\usepackage{subcaption}
\usepackage{amsmath,amssymb}
\usepackage{xcolor}
\usepackage{enumitem}
\usepackage[hidelinks]{hyperref}
\usepackage{cleveref}
\crefname{figure}{Figure}{Figures} % "Figure 2" instead of "fig. 2"
\crefname{table}{Table}{Tables}
\crefname{section}{Section}{Sections}
\crefname{subsection}{Section}{Sections}
\usepackage[nottoc]{tocbibind} % adds References to the table of contents
\usepackage[titles]{tocloft}    % finer control of the table of contents
\setlength{\cftbeforesecskip}{0.5em} % less space above each section entry, so the contents fit on one page
\usepackage{tikz}
\usetikzlibrary{calc,patterns,arrows.meta}

% ---------- Colours ----------
\definecolor{primary}{RGB}{16,64,44}   % dark green
\definecolor{accent}{RGB}{70,160,100}  % green

% All section, subsection and subsubsection headings in dark green
\usepackage{sectsty}
\allsectionsfont{\color{primary}}

% Placeholder for unfinished text (remove before submission)
\newcommand{\todo}[1]{\textcolor{red}{\textbf{TODO:} #1}}

% Indicative page targets from the task description.
% Switch to \showtargetsfalse before submission to hide them all.
\newif\ifshowtargets
\showtargetstrue
\newcommand{\pagetarget}[1]{%
  \ifshowtargets
    \noindent{\small\itshape\textcolor{gray}{Target length: #1 pages (indicative)}}\par\medskip
  \fi}

\begin{document}

% ---------- Title page ----------
\newgeometry{left=4cm,right=2cm,top=2.5cm,bottom=2.5cm} % shifted right, clear of the band
\begin{titlepage}
\begin{tikzpicture}[remember picture,overlay]
  % Dark green band along the left edge
  \fill[primary] (current page.north west) rectangle ([xshift=1.8cm]current page.south west);
  % Thin accent stripe
  \fill[accent] ([xshift=1.95cm]current page.north west) rectangle ([xshift=2.05cm]current page.south west);
\end{tikzpicture}

\centering
\vspace*{\fill}

{\color{accent}\large\scshape TDT4259 -- Group Project\par}
\vspace{1.5em}
{\color{primary}\rule{0.6\textwidth}{1pt}\par}
\vspace{1.5em}
{\color{primary}\huge\bfseries Day-Ahead Hourly\\[0.3em]
 Electricity Consumption Forecasting\par}
\vspace{0.8em}
{\Large\color{primary} for a Nordic Electricity Retailer\par}
\vspace{0.5em}
{\large\itshape Aneo case\par}
\vspace{1.5em}
{\color{primary}\rule{0.6\textwidth}{1pt}\par}

\vspace{3em}
{\large
Nils Henrik Hoelfeldt Lund \\[0.3em]
Jørgen August Bolseth \\[0.3em]
Camilla Eriksen \\[0.3em]
Truls Elverum \\[0.3em]
Thomas Myrseth \\[0.3em]
Niklas Unneland \par}

\vspace{3em}
{\large Norwegian University of Science and Technology (NTNU)\\[0.3em]
October 2026\par}

\vspace*{\fill}
\end{titlepage}
\restoregeometry

% ---------- Front matter ----------
\pagenumbering{roman}
\begingroup
\small % slightly smaller so the contents fit on one page
\tableofcontents
\endgroup
\newpage


\pagenumbering{arabic}

\section{Problem Definition and Business Understanding}
\label{sec:problem}
\subsection{Context}

Aneo is a Nordic renewable energy company headquartered in Trondheim. The company was established in 2022, and is owned by TrønderEnergi and the investment fund HitecVision \cite{aneo_sweden}. In addition to producing wind and hydropower, Aneo offers energy market services to other actors in the power sector. Through its Energy Management unit, Aneo helps other companies take part in the power markets in Norway, Sweden and Finland. By using data-driven models for forecasting, optimization and trading, Aneo provides decision support to retailers in the power market, in this case at a fixed price \cite{aneo_markets}. One such service is forecasting electricity consumption for energy retailers, which is the focus of this project.

\subsection{Problem and Decision to be Supported}
\label{sec:decision}

The problem is to produce a reliable forecast of the retailer's hourly electricity consumption for the next day. That makes a total of 24 values for each of the six price areas. The forecast is issued every day at around 10:00 and can only use information that is available at that time. Because measured consumption data are delivered with a delay of approximately five days, the most recent consumption data available are from five days before the forecast is made (see \cref{fig:forecast-timeline}). The forecast supports the retailer's decision on how much electricity to plan and nominate for each hour and price area. Deviation between the forecast and actual consumption results in an imbalance, which must be settled at prices that can differ substantially from the spot price.

\begin{figure}[H]
\centering
\begin{tikzpicture}[
  declare function={
    W = 14.5;                                     % chart width
    D = W/7;                                      % width of one day
    hr(\x) = 24*mod(\x,D)/D;                      % hour of day
    dw(\x) = floor(\x/D);                         % day index (0 = 13 Sep)
    we(\x) = ifthenelse(dw(\x)==3,1,ifthenelse(dw(\x)==4,1,0)); % 16--17 Sep = weekend
    mp(\x) = 8.5 + 2*we(\x);                      % morning peak later on weekends
    f(\x)  = 1.0 + 0.10*sin(360*\x/W)
             + (1-0.25*we(\x))*( 0.55*exp(-(hr(\x)-mp(\x))*(hr(\x)-mp(\x))/6)
                               + 0.70*exp(-(hr(\x)-18)*(hr(\x)-18)/10)
                               + 0.25*exp(-(hr(\x)-13)*(hr(\x)-13)/20) );
    g(\x)  = f(\x) + 0.07*sin(360*\x/(D/3)) - 0.03;  % illustrative forecast
    xi     = 5*D + 10*D/24;                       % forecast issued 18 Sep 10:00
  }]
  % Background regions
  \fill[gray!12] ({D},0) rectangle (xi,2.3);
  \fill[pattern=north east lines, pattern color=gray!35] (xi,0) rectangle ({6*D},2.3);
  \fill[primary!12] ({6*D},0) rectangle ({7*D},2.3);
  \node[font=\tiny\sffamily, text=gray!70!black, align=center, fill=white, inner sep=1pt]
    at ({(xi+6*D)/2},0.35) {not yet\\occurred};
  % Region brackets and labels
  \begin{scope}[font=\scriptsize\sffamily]
    \draw[accent, thin, |-|] (0.05,2.5) -- ({D-0.05},2.5);
    \node[anchor=south, text=accent!80!black, align=center] at ({0.5*D},2.55) {Data\\available};
    \draw[gray, thin, |-|] ({D+0.05},2.5) -- (xi,2.5);
    \node[anchor=south, text=gray!80!black] at ({(D+xi)/2},2.55)
      {5-day delay: consumption occurred, data not yet available};
    \draw[primary, thin, |-|] ({6*D+0.05},2.5) -- ({7*D-0.05},2.5);
    \node[anchor=south, text=primary, font=\scriptsize\sffamily\bfseries, align=center]
      at ({6.5*D},2.55) {Predict:\\next day, 24 h};
  \end{scope}
  % Forecast issue time
  \draw[primary, densely dashed, thin] (xi,0) -- (xi,2.3);
  \node[anchor=south east, font=\scriptsize\sffamily, text=primary, fill=gray!12, inner sep=1.5pt]
    at ({xi-0.05},1.98) {Forecast made 18 Sep, 10:00};
  % Light horizontal grid
  \foreach \y in {0.7,1.4}
    \draw[accent, opacity=0.25, densely dotted] (0,\y) -- (W,\y);
  % Consumption that has occurred but is not yet available
  \draw[gray!60, line width=0.8pt, densely dashed]
    plot[domain={D}:xi, samples=300] (\x,{f(\x)});
  % Consumption available, with hourly markers
  \draw[accent, line width=1.1pt] plot[domain=0:{D}, samples=100] (\x,{f(\x)});
  \foreach \k in {0,...,24}
    \fill[accent] ({\k*D/24},{f(\k*D/24)}) circle (0.8pt);
  % Prediction interval and forecast for the target day
  \fill[accent, opacity=0.25]
    plot[domain={6*D}:{7*D}, samples=100] (\x,{g(\x)+0.15})
    -- plot[domain={7*D}:{6*D}, samples=100] (\x,{g(\x)-0.15}) -- cycle;
  \draw[primary, line width=1.1pt]
    plot[domain={6*D}:{7*D}, samples=100] (\x,{g(\x)});
  % Axes
  \draw[primary, line width=0.6pt, ->] (0,0) -- ({W+0.3},0);
  \draw[primary, line width=0.6pt, ->] (0,0) -- (0,2.4);
  \node[anchor=east, font=\scriptsize\sffamily, text=primary] at (-0.05,2.3) {MW};
  \foreach \i in {0,...,7}
    \draw[primary, line width=0.5pt] ({\i*D},0) -- ({\i*D},-0.1);
  \foreach \d [count=\i from 0] in {13 Sep,14 Sep,15 Sep,16 Sep,17 Sep,18 Sep,19 Sep}
    \node[font=\scriptsize\sffamily, text=primary!80] at ({(\i+0.5)*D},-0.3) {\d};
  % Legend
  \begin{scope}[font=\scriptsize\sffamily]
    \draw[accent, line width=1.1pt] (0.3,3.5) -- (0.8,3.5);
    \fill[accent] (0.55,3.5) circle (0.8pt);
    \node[anchor=west, text=primary] (l1) at (0.85,3.5) {Actual (available)};
    \draw[gray!60, line width=0.8pt, densely dashed] ($(l1.east)+(0.3,0)$) -- ++(0.5,0)
      node[anchor=west, text=primary] (l2) {Actual (not yet available)};
    \draw[primary, line width=1.1pt] ($(l2.east)+(0.3,0)$) -- ++(0.5,0)
      node[anchor=west, text=primary] (l3) {Forecast};
    \fill[accent, opacity=0.35] ($(l3.east)+(0.3,-0.08)$) rectangle ++(0.5,0.16);
    \node[anchor=west, text=primary] at ($(l3.east)+(0.85,0)$) {Prediction interval};
  \end{scope}
\end{tikzpicture}
\caption{Information available when the forecast is made, illustrated with the example from the task description (synthetic curves). On 18 September at 10:00, measured consumption is only available up to 13 September because of the approximately 5-day data delay, and the remainder of 18 September has not yet occurred. The forecast must cover all 24 hours of 19 September, using historical consumption up to 13 September, the weather forecast for 19 September and calendar information.}
\label{fig:forecast-timeline}
\end{figure}

\subsection{Stakeholders}

The main stakeholders are:

\begin{description}[leftmargin=0pt, itemsep=3pt, parsep=0pt, topsep=4pt]
    \item[Aneo] Provides next-day hourly consumption forecasts for the retailer's six price areas.
    \item[Electricity retailer] Buys electricity in the day-ahead market and sells it to its customers with a mark-up.
    \item[End customers] Buy electricity as needed from their chosen electricity retailer.
    \item[Metering data hub (Elhub, Datahub)] Delivers metered consumption with a delay of about five days.
    \item[TSOs and eSett (Statnett, Fingrid)] Keep the system in balance and settle the retailer's imbalances.
\end{description}

\subsection{Value Chain and Objectives}
\label{sec:value-chain}

The forecasting service connects several actors, as shown in \cref{fig:actors}. Weather forecast providers supply Aneo with critical information such as temperature, humidity and wind, while metered consumption reaches Aneo through the metering data hubs with a delay of about five days. Based on this, Aneo delivers next-day hourly forecasts to the retailer, which uses them to nominate its expected demand in the day-ahead market. Any deviation between nominated and actual consumption is settled as an imbalance by the TSOs and eSett, and the cost is borne by the retailer.

\begin{figure}[H]
\centering
\begin{tikzpicture}[
  font=\small\sffamily,
  >={Stealth[length=2.2mm]},
  box/.style={draw=primary, rounded corners=3pt, align=center, inner sep=4pt, line width=0.6pt},
  aneo/.style={box, fill=primary, text=white, minimum width=4.2cm, minimum height=1.1cm, font=\small\sffamily\bfseries},
  retail/.style={box, fill=accent!25, text=primary, minimum width=4.2cm, minimum height=1.1cm, font=\small\sffamily\bfseries},
  ext/.style={box, draw=gray!60, fill=gray!8, text=gray!20!black, minimum width=2.9cm, minimum height=0.95cm, font=\footnotesize\sffamily},
  area/.style={box, fill=primary!8, text=primary, minimum width=1.5cm, minimum height=0.7cm, font=\footnotesize\sffamily\bfseries},
  lbl/.style={font=\scriptsize\sffamily, text=primary, align=center},
  flow/.style={->, primary, line width=0.8pt},
  money/.style={->, accent!80!black, line width=0.8pt},
  data/.style={->, gray!70, line width=0.8pt, densely dashed},
]
  % Main actors
  \node[aneo] (aneo) at (0,5) {Aneo\\[-1pt]{\footnotesize\mdseries forecasting service}};
  \node[retail] (ret) at (0,2.6) {Electricity retailer\\[-1pt]{\footnotesize\mdseries balance planning}};
  \node[ext] (wx)  at (-6.5,6.5) {Weather forecasts};
  \node[ext] (hub) at (-6.5,2.6) {Metering data hub\\(Elhub / Datahub)};
  \node[ext] (mkt) at (6.5,3.15) {Power market\\(day-ahead)};
  \node[ext] (tso) at (6.5,2.05) {TSOs and eSett\\(Statnett, Fingrid)};

  % Price areas and customers
  \foreach \a [count=\i from 0] in {NO1,NO2,NO3,NO4,NO5,FI}
    \node[area] (a\i) at ({-4.75+\i*1.9},0.6) {\a};
  \node[box, draw=primary!40, fill=white, text=primary, minimum width=11cm, minimum height=0.75cm,
        font=\footnotesize\sffamily] (cust) at (0,-0.9)
        {End customers in each price area (households and industry)};
  \foreach \i in {0,...,5} {
    \draw[primary!50, line width=0.6pt] (ret.south) -- (a\i.north);
    \draw[primary!40, line width=0.6pt] (a\i.south) -- (a\i.south |- cust.north);
  }
  \node[lbl, anchor=east, text=primary!70] at ([xshift=-0.15cm]a0.west) {Price\\areas};

  % Flows between Aneo and retailer
  \draw[flow] ([xshift=-0.7cm]aneo.south) -- node[lbl, left, xshift=-2pt]
    {Hourly forecast for next day\\24\,h $\times$ 6 areas, by 10:00} ([xshift=-0.7cm]ret.north);
  \draw[money] ([xshift=0.7cm]ret.north) -- node[lbl, right, xshift=2pt, text=accent!60!black]
    {Fixed contract fee} ([xshift=0.7cm]aneo.south);

  % Inputs to Aneo
  \draw[flow] (wx.east) -| node[lbl, above, pos=0.35] {forecast for next day} ([xshift=-1.2cm]aneo.north);
  \draw[data] (cust.west) -| (hub.south);
  \draw[data] (hub.north) |- node[lbl, above, pos=0.75, text=gray!50!black]
    {metered consumption\\($\approx$\,5-day delay)} (aneo.west);

  % Retailer and market
  \draw[flow] ([yshift=0.3cm]ret.east) -- node[lbl, above] {Nomination}
    ([yshift=-0.25cm]mkt.west);
  \draw[money] ([yshift=0.25cm]tso.west) -- node[lbl, below, text=accent!60!black] {Imbalance cost}
    ([yshift=-0.3cm]ret.east);

\end{tikzpicture}
\caption{Actors and flows in the case. Aneo delivers next-day consumption forecasts to the retailer, which uses them to nominate demand and bears the imbalance cost.}
\label{fig:actors}
\end{figure}

\noindent The quality of the input data is crucial, since a predictive model is only as good as the data it is based on. Errors in the weather forecasts propagate through the whole chain: Aneo's predictions become less accurate, which means that the electricity retailer must settle the resulting deficit or surplus at less favourable imbalance prices, reducing its profit.

\medskip
\noindent Errors can also come from the consumption data, for example missing or delayed meter values, or from changes in the retailer's customer portfolio that the model does not capture. Aneo can also be affected by a poor predictive model: even if the weather forecasts are accurate, the forecasts delivered to the retailer will be wrong if the model uses this information incorrectly. An example could be failing to rank the importance of each source, leading to a wrong combination or formula. Customers are also affected, although only indirectly: if imbalance costs remain high over time, the retailer may pass some of them on through higher prices, which in turn can lead customers to switch to another retailer.

\medskip
\noinde Since the retailer pays Aneo a fixed fee, Aneo does not earn or lose money directly from each imbalance. Instead, the value of the service lies in how it affects the retailer, as illustrated in \cref{fig:value-chain}. More accurate forecasts reduce the retailer's imbalance and financial exposure, which increases the perceived value of the service and the retailer's willingness to pay for it. For Aneo, this ultimately translates into customer satisfaction and a higher likelihood of contract renewal.

\begin{figure}[H]
\centering
\begin{tikzpicture}[
  >={Stealth[length=2.2mm]},
  chain/.style={draw=accent, rounded corners=3pt, fill=white, text=primary, align=center,
                inner sep=4pt, line width=0.6pt, minimum width=2.6cm, minimum height=1cm,
                font=\footnotesize\sffamily},
  flow/.style={->, primary, line width=0.8pt},
]
  \foreach \t [count=\i from 0] in {Better\\forecast, Smaller\\imbalance, Lower financial\\exposure for retailer,
                                    Higher willingness\\to pay Aneo, Satisfaction and\\contract renewal}
    \node[chain] (c\i) at ({\i*3.15},0) {\t};
  \foreach \i [evaluate=\i as \j using int(\i+1)] in {0,...,3}
    \draw[flow] (c\i.east) -- (c\j.west);
\end{tikzpicture}
\caption{Value chain for Aneo. Because Aneo is paid a fixed fee, forecast quality creates value indirectly, through the retailer's willingness to pay for the service and the likelihood of contract renewal.}
\label{fig:value-chain}
\end{figure}

Based on this, the project has three main objectives:

\begin{description}[leftmargin=0pt, itemsep=2pt]
    \item[Technical] Produce accurate and unbiased hourly forecasts that outperform a simple baseline in all six price areas.
    \item[Business] Reduce the retailer's total imbalance and the frequency of large forecast errors.
    \item[Operational] Deliver the forecasts reliably before 10:00 every day, using only information available at that time.
\end{description}

\noindent The measurable criteria used to evaluate these objectives are defined in \cref{sec:success}.

\subsection{Success Criteria}
\label{sec:success}

The objectives in \cref{sec:value-chain} are evaluated on a held-out test period that the models have not seen during training or tuning (see \cref{sec:validation}).

\begin{description}[leftmargin=0pt, itemsep=2pt]
    \item[Technical] Lower mean absolute error (MAE) than the same-hour-last-week baseline in all six price areas, and a bias close to zero. MAE is the main measure because it penalises over- and underforecasting equally and grows linearly with the error, in line with how imbalance costs arise:
    \begin{equation}
        \text{MAE} = \frac{1}{n}\sum_{t=1}^{n} \left| y_t - \hat{y}_t \right|,
        \label{eq:mae}
    \end{equation}
    where $y_t$ is the actual consumption, $\hat{y}_t$ the forecast and $n$ the number of hours.
    \item[Business] Lower total imbalance (MWh) and fewer hours with large errors than the baseline, consistently across price areas and seasons. Specific target levels would in practice be agreed with Aneo and the retailer, for example in a service-level agreement.
    \item[Operational] The forecast is produced before 10:00 every day, using only data available at that time.
\end{description}

\subsection{Constraints and Assumptions}

The main constraints are that the forecast is issued at around 10:00 for the next day, that consumption data are only available up to five days earlier, and that only temperature forecasts and calendar information are available as additional inputs (see \cref{sec:decision}). We further assume that:

\begin{itemize}[itemsep=0pt, topsep=4pt]
    \item the retailer pays Aneo a fixed fee, and over- and underforecasting are equally costly;
    \item imbalance costs scale approximately linearly with the forecast error, so the full settlement mechanism messnot modelled;
    \item the temperature values are forecasts available at the time the consumption forecast is made;
    \item Aneo's current forecasting method is unknown, so a simple baseline is used as the reference.
\end{itemize}
\newpage

% =====================================================================
\section{Data Understanding and Background}
\label{sec:data}
\pagetarget{3--4}

\subsection{Domain Background}
\todo{Drivers of Nordic electricity consumption: electric heating $\Rightarrow$ strong temperature dependence and winter peaks; daily and weekly patterns; holidays; differences between price areas; FI vs. NO. Brief on imbalance settlement (single price model, 15-min MTU since 2025 -- check and simplify).}

The electricity market divides Norway into 5 different price-regions, while Finland is kept as one big price region. Supply and Demand within each region combined with possible trade between price regions, depending on transmission capacity dictates a market price for each delivery interval (currently one hour, but will move towards 15-minutes intervals), the day ahead. The retailer's customers are based inside the price region, but it's important to note that the retailers customers only account for a subset of the consumers inside the price region. Therefore consumption of the customers can be dependent on a different mix of variables than that of the price region as a whole.

One of the biggest consumers of electricity is that of heating. Norway and Finland are both located in cold climates where heating is a necessity, especially in the winter months. Especially households have a higher consumption of electricity in the winter due to heating, thus temperature is a key input-variable for forecasting demand. Consumption also follows daily patterns. When the population wakes up, lights are turned on, showers use warm water that needs to be replenished etc and drives up demand. During the working day, industry is driving the demand, while households are driving the demand again in the evening cooking, doing laundry etc. During the night activity falls and so does the power consumption.

The retailer submits bids in the day-ahead market on Nord Pool for the price regions in which it operates. Accepted bids are binding, although the retailer can adjust its purchases through intraday trading before delivery. Since electricity production and consumption must remain balanced, the TSOs use balancing markets to address differences between planned and actual production and consumption. The retailer bears financial responsibility for deviations between its contracted purchases and its customers’ actual consumption, either directly as a balance responsible party or through an agreement with one. These deviations are settled at the imbalance price, which can differ from the day-ahead price. Forecast errors can therefore cause financial losses for the retailer.

Public holidays may also affect electricity consumption by changing household routines and business activity. However, our dataset covers only one year, providing limited examples from which to learn holiday effects. These effects may therefore be difficult to distinguish from other influences, such as temperature and day of the week.





\subsection{Dataset Description}
\todo{\texttt{consumption\_temp.csv}: columns Time (UTC), Location, Consumption (MW, hourly average $\approx$ MWh per hour), Temperature (forecast, $^\circ$C). Time span, number of rows per location, mapping Location $\rightarrow$ price area.}
% Table: location | price area | period | mean/min/max consumption


The dataset \texttt{consumption\_temp.csv} contains 49,494 hourly
observations across six locations in Norway and Finland. Each row
contains four variables: \texttt{time}, the timestamp in UTC;
\texttt{location}, the location identifier; \texttt{consumption},
the average electricity demand during the hour in MW; and
\texttt{temperature}, the forecast temperature in $^\circ$C.
An average demand of 1 MW over one hour corresponds to an energy
consumption of 1 MWh.

The Norwegian locations cover the period from 7 April 2022 at
21:00 to 2 April 2023 at 21:00, with 8,641 observations per
location. Finland, labelled \texttt{helsingfors} in the dataset,
starts on 14 July 2022 at 21:00 and ends at the same time as the
Norwegian series, giving 6,289 observations.

Table~\ref{tab:dataset_locations} shows the location identifiers
and their corresponding electricity price areas. The consumption
series represent the retailer's customers associated with each
location, rather than total consumption in the price area.

\begin{table}[htbp]
    \centering
    \caption{Locations, corresponding price areas and observation counts.}
    \label{tab:dataset_locations}
    \begin{tabular}{llr}
        \hline
        Location & Price area & Observations \\
        \hline
        Oslo       & NO1 & 8,641 \\
        Stavanger  & NO2 & 8,641 \\
        Trondheim  & NO3 & 8,641 \\
        Troms{\o}  & NO4 & 8,641 \\
        Bergen     & NO5 & 8,641 \\
        Helsingfors & FI & 6,289 \\
        \hline
        Total & & 49,494 \\
        \hline
    \end{tabular}
\end{table}

There are no missing values or duplicate timestamp--location
pairs in the file. Each location has an uninterrupted hourly
series within its available period, although the Finnish series
covers a shorter period.

\subsection{Important Note on Interpretation}
\todo{Consumption is only the retailer's portfolio in each area, not total area demand. City size is irrelevant; portfolio size changes over time (customer gains/losses) may cause level shifts.}

\subsection{Exploratory Data Analysis}
\todo{Time series per area; daily/weekly/yearly profiles; consumption vs. temperature (likely non-linear, heating threshold); holiday effects; autocorrelation at lags $\geq$ 144 h (e.g. 168 h).}

\subsection{Data Quality Issues}
\todo{Missing hours, duplicates, DST/UTC issues, outliers, structural breaks (portfolio changes), series starting at different dates, temperature being a forecast with its own error.}

\subsection{Possible External Data}
\todo{Optional: Norwegian/Finnish holiday calendars; Nord Pool spot prices and eSett/Nord Pool imbalance prices for cost estimation (business evaluation, not necessarily features).}

\subsection{Assumptions and Limitations}
\todo{E.g. temperature forecast quality assumed similar in production; one representative location per area; no customer-level data.}

% =====================================================================
% Ansvarlig: Thomas & Truls
\section{Methods, Analysis and Technical Evaluation}
\label{sec:methods}
\pagetarget{4--6}

\subsection{Reproducing the Forecasting Situation}
\todo{Explain precisely what information is available at forecast time and how this is enforced in feature construction (no lags shorter than 144 h for the earliest target hour; check lags hour by hour). Illustrate with the 18 Sep $\rightarrow$ 19 Sep example.}
% Tip: a timeline figure here is very effective.

\subsection{Data Preparation and Feature Engineering}
\todo{Cleaning; conversion UTC $\rightarrow$ local time for calendar features; features: hour, weekday, weekend, month/season, holidays, temperature forecast (+ non-linear terms, heating degree, rolling temp), consumption lags $\geq$ 6 days (e.g. 168 h, 336 h), rolling statistics computed only from available data.}

\subsection{Baselines}
\todo{(1) Same hour one week earlier (required benchmark). (2) Possibly mean of same hour/weekday last N available weeks. (3) Simple linear regression on temperature + calendar.}

\subsection{Models}
\todo{E.g. linear/ridge regression, gradient boosting (LightGBM/XGBoost), possibly SARIMAX or simple NN. One global model vs. one model per area -- justify choice.}

\subsection{Validation Approach}
\label{sec:validation}
\todo{Chronological split (train / validation / test); rolling-origin (expanding window) backtest that simulates daily issuing at 10:00; hyperparameter tuning only on validation period; no random shuffling.}

\subsection{Evaluation Metrics}
\todo{MAE (main, symmetric loss), RMSE, MAPE/nMAE for comparing areas of different size, bias (mean error), error by hour of day, error in high-consumption periods (e.g. top 10\% hours / winter).}

\subsection{Results}
\todo{Main comparison table and figures: predicted vs. actual for sample weeks, error by hour, error by area and month.}
\begin{table}[H]
\centering
\caption{Test-set performance per model (all areas). \todo{fill in}}
\label{tab:results}
\begin{tabular}{lcccc}
\toprule
Model & MAE [MW] & RMSE [MW] & nMAE [\%] & Bias [MW] \\
\midrule
Same hour last week  & -- & -- & -- & -- \\
Linear regression    & -- & -- & -- & -- \\
Gradient boosting    & -- & -- & -- & -- \\
\bottomrule
\end{tabular}
\end{table}

\subsection{Error Analysis and Robustness}
\todo{Performance per price area, season, holidays, cold spells; feature importance; sensitivity to temperature forecast error; behaviour at portfolio level shifts.}

\subsection{Technical Limitations}
\todo{Limited history, single weather variable, one location per area, test period representativeness.}

% =====================================================================
% Ansvarlig: Thomas & Truls
\section{Business Evaluation, Value and Recommendations}
\label{sec:business}
\pagetarget{3--5}

\subsection{From Forecast Error to Imbalance}
\label{sec:imbalance}
\todo{Introduce: since each value is an hourly average in MW, the absolute forecast error for one hour equals the imbalance in MWh. Then define:}
\begin{align}
    I &= \sum_{a=1}^{6}\sum_{t=1}^{n} \left| y_{a,t} - \hat{y}_{a,t} \right| \cdot 1\,\text{h}, \label{eq:imbalance} \\
    \Delta I &= \frac{I_{\text{baseline}} - I_{\text{model}}}{I_{\text{baseline}}} \times 100\,\%, \label{eq:reduction}
\end{align}
where $I$ is the total absolute imbalance in MWh, summed over all price areas $a$ and hours $t$, and $\Delta I$ is the percentage reduction in imbalance compared with the baseline. \todo{Define large errors, e.g. errors above the 90th percentile of the baseline's absolute errors.}

\subsection{Business KPIs}
\todo{Total absolute imbalance (MWh) per area and in total; \% reduction vs. baseline; frequency of large errors (e.g. $>$ X\% of consumption); \% of hours within acceptable error band; reliability across areas and periods (worst month/area).}
\begin{table}[H]
\centering
\caption{Business KPIs on the test period. \todo{fill in}}
\label{tab:business}
\begin{tabular}{lccc}
\toprule
KPI & Baseline & Best model & Change \\
\midrule
Total abs. imbalance [MWh]      & -- & -- & -- \\
Hours with large error [\%]     & -- & -- & -- \\
Hours within tolerance [\%]     & -- & -- & -- \\
Est. imbalance cost [EUR]       & -- & -- & -- \\
\bottomrule
\end{tabular}
\end{table}

\subsection{Estimated Imbalance Cost (optional)}
\todo{If price data is found: cost$_t \approx |\hat{y}_t - y_t| \cdot |p^{\text{imb}}_t - p^{\text{spot}}_t|$. State assumptions and simplifications clearly. Otherwise use a sensitivity range of price differences.}

\subsection{Value for the Retailer and for Aneo}
\todo{Retailer: reduced financial exposure and risk. Aneo: higher service value $\rightarrow$ satisfaction, retention, contract renewal. Discuss how this could be measured (renewal rate, satisfaction surveys, accuracy SLA).}

\subsection{Costs, Risks and Trade-offs}
\todo{Development and operating cost vs. value; model complexity vs. interpretability/maintenance; risk of large errors in extreme weather; dependence on weather provider and data delivery.}

\subsection{Recommendations}
\todo{Which model to deploy (possibly per area), whether it is good enough to justify continued service, suggested accuracy SLA, what to improve next (more weather variables, shorter data delay, etc.).}

% =====================================================================
% Ansvarlig: Camilla & Jørgen
\section{Deployment and Production Readiness}
\label{sec:deployment}
\pagetarget{3--5}

\subsection{Operational Workflow}
\todo{Daily batch job: fetch latest available consumption (D$-$5) and weather forecast for D+1 $\rightarrow$ build features $\rightarrow$ predict 24$\times$6 values $\rightarrow$ validate output $\rightarrow$ deliver to retailer well before day-ahead gate closure (12:00 CET).}
% Tip: architecture/workflow diagram

\subsection{System Components}
\todo{Data ingestion (meter data, weather API), feature pipeline, model registry, scheduler (e.g. cron/Airflow), output delivery (API/file), storage of forecasts and actuals for evaluation.}

\subsection{Fallback and Robustness}
\todo{If data or model fails: fall back to baseline (same hour last week) so a forecast is always delivered on time; sanity checks on output (non-negative, plausible range).}

\subsection{Testing and Validation before Deployment}
\todo{Unit tests of feature code (especially leakage tests: no data newer than D$-$5), data validation (schema, missing hours, DST), backtesting, shadow mode in parallel with existing process, acceptance criteria vs. baseline.}

\subsection{Reproducibility and Versioning}
\todo{Git for code, versioned data snapshots, model registry with version + training period, fixed environments (requirements/Docker), logged forecasts.}

\subsection{Security and Compliance}
\todo{Consumption data is commercially sensitive; access control; agreements between Aneo and retailer; aggregated data limits GDPR concerns.}

\subsection{Scalability and Staged Deployment}
\todo{Pilot in one or two areas $\rightarrow$ shadow mode $\rightarrow$ full rollout; easy to add new areas/retailers; rollback plan.}

% =====================================================================
% Ansvarlig: Camilla & Jørgen
\section{Monitoring and Maintenance}
\label{sec:monitoring}
\pagetarget{2--4}

\subsection{Data Monitoring}
\todo{Delivery delay of consumption data (does it stay $\sim$5 days?), missing or late weather forecasts, missing hours, distribution drift in temperature/consumption.}

\subsection{Forecast Performance Monitoring}
\todo{Rolling MAE/nMAE and bias per area (e.g. 7- and 30-day windows) vs. baseline; error by hour; note that actuals arrive with 5-day delay, so monitoring is delayed too.}

\subsection{Business KPI Monitoring}
\todo{Monthly total imbalance (MWh) and estimated cost vs. baseline; reporting to retailer; retailer satisfaction and contract status.}

\subsection{Failure Modes and Alerts}
\begin{table}[H]
\centering
\caption{Failure modes, indicators and actions. \todo{fill in}}
\label{tab:failures}
\begin{tabularx}{\textwidth}{XXXX}
\toprule
Failure mode & Indicator / alert & Action & Owner \\
\midrule
Missing weather forecast & No data at 09:30 & Use fallback & Aneo ops \\
Portfolio change (customers gained/lost) & Persistent bias & Adjust level / retrain & Aneo + retailer \\
Model degradation / drift & Rolling MAE $>$ threshold & Investigate, retrain & Aneo forecast team \\
Extreme weather & Temp. outside training range & Manual review & Aneo forecast team \\
\bottomrule
\end{tabularx}
\end{table}

\subsection{Responsibilities}
\todo{Aneo forecast team (model), Aneo ops/IT (pipeline), retailer (data delivery, feedback, informing about portfolio changes), account manager (business KPIs, relationship).}

\subsection{Maintenance Actions}
\todo{Scheduled retraining (e.g. monthly/seasonal) and triggered retraining on drift; yearly holiday calendar update; periodic re-evaluation against baseline; documentation and model changelog.}

% =====================================================================
\section{Conclusion}
\label{sec:conclusion}
\todo{Summarise the chain: problem $\rightarrow$ data $\rightarrow$ analysis $\rightarrow$ evidence $\rightarrow$ decision/value $\rightarrow$ deployment $\rightarrow$ monitoring. Answer the business question: is the solution reliable and valuable enough? Limitations and future work.}

% =====================================================================
% References (entries are in references.bib)
\bibliographystyle{plain}
\bibliography{references}

\appendix
\section{Additional Figures and Tables}
\todo{Per-area results, extra EDA plots, hyperparameters.}

\end{document}