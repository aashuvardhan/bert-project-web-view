import logo from './logo.svg';
import './App.css';
import React,{useState,useEffect} from 'react';
import { Form } from './Components/Form/Form';
import {Card} from './Components/Card/Card'
import LoadingSpinner from './Components/Form/LoadingSpinner';

function App() {

  const [data,setData]=useState([])
  const [isLoading,setIsLoading]=useState(null)
  const [userInput,setUserInput]=useState('')

  useEffect(()=>{
    fetch('/data').then((response)=>{
      
        if(response.ok){
            return response.json()
        }
    }).then((data)=>{
      //console.log(data)
      setData(data[0]['content'])
    })
  },[])

  
  const formHandler=(content)=>{
    setUserInput(content)
    postRequestSender(content)
  }



  const postRequestSender=(value)=>{
        
        setIsLoading(true)
        fetch('/',{
            method:'POST',
            body:JSON.stringify({
                content:value
            }),
            headers:{
                "Content-type":"application/json; charset=UTF-8"
            }
        }).then(res=>res.json())
            .then(message=>{
                
                setIsLoading(false)
                getLatestData()
            })
  }

  const getLatestData=()=>{
    fetch('/data').then((response)=>{
        if(response.ok){
            return response.json()
        }
    }).then((data)=>{
      setData(data[0]['content'])
    })
}

  return (
    <div className="App">
      <h1>Welcome to the Missing word prediction Page.</h1>
      <h4>You can put '---' in the entered sentence to predict the missing word.</h4>
      <Form formHandler={formHandler}/>
      <div>{isLoading==null && <p>Enter your First Sentence to predict top 5 suitable missing words.</p>}</div>
      <div>{isLoading && <p>We are processing the results. Please be patient.</p>}</div>
      <div>{isLoading && <LoadingSpinner /> }</div>
      <div>{isLoading==false && <p>Your Input  :-   {userInput}</p>}</div>
      <div>{isLoading==false && <p>Prediction are:-</p>}</div>
      {isLoading==false && <Card data={data}/>}
    </div>
  );
}

export default App;
