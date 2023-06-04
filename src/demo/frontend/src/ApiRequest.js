import axios from 'axios';

const ApiRequest = async (text) => {
  try {
    const response = await axios.post('http://localhost:3000/predict/', { text });
    return response.data;
  } catch (error) {
    console.error(error);
  }
};

export default ApiRequest;