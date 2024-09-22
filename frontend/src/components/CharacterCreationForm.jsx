import React, { useState, useRef, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardFooter } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Textarea } from '@/components/ui/textarea';
import { Button } from '@/components/ui/button';
import { Switch } from '@/components/ui/switch';
import { Label } from '@/components/ui/label';
import { Send } from 'lucide-react';

const CharacterCreationForm = () => {
  const [character, setCharacter] = useState({
    name: '',
    tagline: '',
    description: '',
    greeting: '',
    isPublic: true,
    useFinetuning: false
  });
  const [showChat, setShowChat] = useState(false);
  const [chatMessages, setChatMessages] = useState([]);
  const [inputMessage, setInputMessage] = useState('');
  const chatContainerRef = useRef(null);

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setCharacter(prev => ({ ...prev, [name]: value }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    console.log('Submitting character:', character);
    await new Promise(resolve => setTimeout(resolve, 1000));
    setShowChat(true);
    setChatMessages([{ role: 'assistant', content: character.greeting || `Hello! I'm ${character.name}. How can I assist you today?` }]);
  };

  const sendMessage = async () => {
    if (!inputMessage.trim()) return;

    const userMessage = { role: 'user', content: inputMessage };
    setChatMessages(prev => [...prev, userMessage]);
    setInputMessage('');

    const aiResponse = await getAIResponse(inputMessage, character);
    const assistantMessage = { role: 'assistant', content: aiResponse };
    setChatMessages(prev => [...prev, assistantMessage]);
  };

  const getAIResponse = async (message, character) => {
    await new Promise(resolve => setTimeout(resolve, 1000));
    if (character.useFinetuning) {
      return `[Fine-tuned ${character.name}]: This would be a response from the fine-tuned model based on your message: "${message}"`;
    } else {
      return `[Contextual ${character.name}]: This would be a response using contextual adaptation with Llama 2, considering your message: "${message}"`;
    }
  };

  useEffect(() => {
    if (chatContainerRef.current) {
      chatContainerRef.current.scrollTop = chatContainerRef.current.scrollHeight;
    }
  }, [chatMessages]);

  if (showChat) {
    return (
      <Card className="w-full max-w-2xl mx-auto h-[600px] flex flex-col bg-gradient-to-br from-purple-50 to-blue-50 shadow-lg">
        <CardHeader className="border-b bg-white bg-opacity-80 backdrop-blur-sm">
          <h2 className="text-2xl font-bold text-purple-700">{character.name}</h2>
          <p className="text-sm text-gray-500">{character.tagline}</p>
        </CardHeader>
        <CardContent className="flex-grow overflow-y-auto p-4" ref={chatContainerRef}>
          <div className="space-y-4">
            {chatMessages.map((msg, index) => (
              <div key={index} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                <div className={`p-3 rounded-lg max-w-[80%] ${
                  msg.role === 'user' 
                    ? 'bg-purple-500 text-white rounded-br-none' 
                    : 'bg-white text-gray-800 rounded-bl-none shadow-md'
                }`}>
                  {msg.content}
                </div>
              </div>
            ))}
          </div>
        </CardContent>
        <CardFooter className="border-t bg-white bg-opacity-80 backdrop-blur-sm">
          <div className="flex w-full items-center space-x-2">
            <Input
              value={inputMessage}
              onChange={(e) => setInputMessage(e.target.value)}
              placeholder="Type your message..."
              className="flex-grow bg-white border-purple-200 focus:border-purple-400 focus:ring-purple-400"
              onKeyPress={(e) => {
                if (e.key === 'Enter') {
                  sendMessage();
                }
              }}
            />
            <Button 
              onClick={sendMessage}
              className="bg-purple-500 hover:bg-purple-600 text-white"
            >
              <Send size={18} />
            </Button>
          </div>
        </CardFooter>
      </Card>
    );
  }

  return (
    <Card className="w-full max-w-2xl mx-auto bg-gradient-to-br from-purple-50 to-blue-50 shadow-lg">
      <CardHeader className="bg-white bg-opacity-80 backdrop-blur-sm">
        <h2 className="text-2xl font-bold text-purple-700">Create a Character</h2>
      </CardHeader>
      <CardContent>
        <form onSubmit={handleSubmit} className="space-y-4">
          <div>
            <Label htmlFor="name" className="text-purple-700">Character Name</Label>
            <Input id="name" name="name" value={character.name} onChange={handleInputChange} placeholder="e.g. Albert Einstein" className="border-purple-200 focus:border-purple-400 focus:ring-purple-400" />
          </div>
          <div>
            <Label htmlFor="tagline" className="text-purple-700">Tagline</Label>
            <Input id="tagline" name="tagline" value={character.tagline} onChange={handleInputChange} placeholder="Add a short tagline of your Character" className="border-purple-200 focus:border-purple-400 focus:ring-purple-400" />
          </div>
          <div>
            <Label htmlFor="description" className="text-purple-700">Description</Label>
            <Textarea id="description" name="description" value={character.description} onChange={handleInputChange} placeholder="How would your Character describe themselves?" className="border-purple-200 focus:border-purple-400 focus:ring-purple-400" />
          </div>
          <div>
            <Label htmlFor="greeting" className="text-purple-700">Greeting</Label>
            <Input id="greeting" name="greeting" value={character.greeting} onChange={handleInputChange} placeholder="e.g. Hello, I am Albert. Ask me anything about my scientific contributions." className="border-purple-200 focus:border-purple-400 focus:ring-purple-400" />
          </div>
          <div className="flex items-center space-x-2">
            <Switch id="isPublic" checked={character.isPublic} onCheckedChange={(checked) => setCharacter(prev => ({ ...prev, isPublic: checked }))} />
            <Label htmlFor="isPublic" className="text-purple-700">Public</Label>
          </div>
          <div className="flex items-center space-x-2">
            <Switch id="useFinetuning" checked={character.useFinetuning} onCheckedChange={(checked) => setCharacter(prev => ({ ...prev, useFinetuning: checked }))} />
            <Label htmlFor="useFinetuning" className="text-purple-700">Use Fine-tuning</Label>
          </div>
        </form>
      </CardContent>
      <CardFooter className="bg-white bg-opacity-80 backdrop-blur-sm">
        <Button onClick={handleSubmit} className="w-full bg-purple-500 hover:bg-purple-600 text-white">Create Character</Button>
      </CardFooter>
    </Card>
  );
};

export default CharacterCreationForm;
