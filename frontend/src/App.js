import React from 'react';
import { ChakraProvider } from '@chakra-ui/react';
import CharacterCreationForm from './components/CharacterCreationForm';

function App() {
  return (
    <ChakraProvider>
      <CharacterCreationForm />
    </ChakraProvider>
  );
}

export default App;
