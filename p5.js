import React, { useState } from "react";
// Child component using Props
function Greeting({ name }) {
  return <h2>Hello, {name}!</h2>;  }
// Parent component using State
function App() {
  const [count, setCount] = useState(0);
  const [name, setName]   = useState("Student");
  return (
    <div>
      <h1>React Demo</h1>
      <Greeting name={name} />
      <input  value={name}
        onChange={(e) => setName(e.target.value)}
        placeholder="Enter name"/>
      <p>Count: {count}</p>
      <button onClick={() => setCount(count + 1)}>Increment</button>
      <button onClick={() => setCount(count - 1)}>Decrement</button>
    </div>
  );   } export default App;
