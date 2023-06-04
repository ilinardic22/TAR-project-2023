import React, { useState, useEffect } from 'react';
import Chart from 'chart.js/auto';
import ApiRequest from './ApiRequest';
import loadingImage from './loadingImg.gif';

const Form = () => {
  const [text, setText] = useState('');
  const [chartData, setChartData] = useState(null);
  const [result, setResult] = useState(null);
  const [isLoading, setIsLoading] = useState(false);

  useEffect(() => {
    if (chartData) {
      chartData.destroy();
    }

    if (result) {
      const labels = Object.keys(result);
      const values = Object.values(result);

      const backgroundColors = values.map((value) => {
        return value >= 0.5 ? 'green' : 'red';
      });

      const ctx = document.getElementById('chart');
      const chart = new Chart(ctx, {
        type: 'bar',
        data: {
          labels: labels,
          datasets: [
            {
              label: 'Value',
              data: values,
              backgroundColor: backgroundColors, // 'rgba(75, 192, 192, 0.2)',
              borderColor: 'rgba(75, 192, 192, 1)',
              borderWidth: 1,
            },
          ]
        },
        options: {
          maintainAspectRatio: false,
          scales: {
            y: {
              beginAtZero: true,
              max: 1,
            },
          },
        },
      });

      setChartData(chart);
    }
  }, [result]);

  const handleSubmit = async (event) => {
    event.preventDefault();
    setIsLoading(true);
    const response = await ApiRequest(text);
    setResult(response);
    setIsLoading(false);
  };

  return (
    <div style={{ width: '30vw', height: '70vh' }}>
      <form onSubmit={handleSubmit}>
        <input
          type="text"
          value={text}
          onChange={(event) => setText(event.target.value)}
        />
        <button type="submit">Predict</button>
      </form>
      {isLoading && (
        <div className="loader-container">
          <img src={loadingImage} alt="Loading" />
        </div>
      )}
      <div style={{ height: 'calc(100% - 40px)' }}>
        <canvas id="chart" style={{ width: '100%', height: '100%' }}></canvas>
      </div>
    </div>
  );
};

export default Form;