import React, { useState } from 'react';
import ApiRequest from './ApiRequest';

const Form = () => {
  const [text, setText] = useState('');
  const [result, setResult] = useState('');

  const handleSubmit = async (event) => {
    event.preventDefault();
    const response = await ApiRequest(text);
    setResult(response);
    return null;
  };

  return (
    <div>
      <form onSubmit={handleSubmit}>
        <input
          type="text"
          value={text}
          onChange={(event) => setText(event.target.value)}
        />
        <button type="submit">Submit</button>
      </form>
      <div>{result}</div>
    </div>
  );
};

export default Form;