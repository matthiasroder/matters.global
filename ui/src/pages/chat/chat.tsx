import { ChatInput } from "@/components/custom/chatinput";
import { PreviewMessage, ThinkingMessage } from "../../components/custom/message";
import { useScrollToBottom } from '@/components/custom/use-scroll-to-bottom';
import { useState, useRef, useEffect } from "react";
import { message } from "../../interfaces/interfaces"
import { Overview } from "@/components/custom/overview";
import { Header } from "@/components/custom/header";
import {v4 as uuidv4} from 'uuid';
import { WEBSOCKET_ENDPOINT } from "@/config";

export function Chat() {
  const [messagesContainerRef, messagesEndRef] = useScrollToBottom<HTMLDivElement>();
  const [messages, setMessages] = useState<message[]>([]);
  const [question, setQuestion] = useState<string>("");
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [connectionStatus, setConnectionStatus] = useState<"connecting" | "connected" | "disconnected" | "error">("connecting");

  const socketRef = useRef<WebSocket | null>(null);
  const messageHandlerRef = useRef<((event: MessageEvent) => void) | null>(null);

  // WebSocket connection management
  useEffect(() => {
    const connectWebSocket = () => {
      try {
        setConnectionStatus("connecting");
        const socket = new WebSocket(WEBSOCKET_ENDPOINT);
        
        socket.onopen = () => {
          console.log("WebSocket connected");
          setConnectionStatus("connected");
        };
        
        socket.onclose = () => {
          console.log("WebSocket disconnected");
          setConnectionStatus("disconnected");
          // Attempt to reconnect after 3 seconds
          setTimeout(connectWebSocket, 3000);
        };
        
        socket.onerror = (error) => {
          console.error("WebSocket error:", error);
          setConnectionStatus("error");
        };
        
        socketRef.current = socket;
      } catch (error) {
        console.error("Failed to create WebSocket:", error);
        setConnectionStatus("error");
      }
    };

    connectWebSocket();

    // Cleanup on unmount
    return () => {
      if (socketRef.current) {
        socketRef.current.close();
      }
    };
  }, []);

  const cleanupMessageHandler = () => {
    if (messageHandlerRef.current && socketRef.current) {
      socketRef.current.removeEventListener("message", messageHandlerRef.current);
      messageHandlerRef.current = null;
    }
  };

async function handleSubmit(text?: string) {
  const socket = socketRef.current;
  if (!socket || socket.readyState !== WebSocket.OPEN || isLoading) return;

  const messageText = text || question;
  setIsLoading(true);
  cleanupMessageHandler();
  
  const traceId = uuidv4();
  setMessages(prev => [...prev, { content: messageText, role: "user", id: traceId }]);
  socket.send(messageText);
  setQuestion("");

  try {
    const messageHandler = (event: MessageEvent) => {
      const data = event.data;

      // Check if this is an end marker
      if (data === "[END]") {
        cleanupMessageHandler();
        setIsLoading(false);
        return;
      }

      // Process regular message content
      setMessages(prev => {
        const lastMessage = prev[prev.length - 1];
        const newContent = lastMessage?.role === "assistant"
          ? lastMessage.content + data
          : data;

        const newMessage = { content: newContent, role: "assistant", id: traceId };
        return lastMessage?.role === "assistant"
          ? [...prev.slice(0, -1), newMessage]
          : [...prev, newMessage];
      });
    };

    messageHandlerRef.current = messageHandler;
    socket.addEventListener("message", messageHandler);
  } catch (error) {
    console.error("WebSocket error:", error);
    setIsLoading(false);
  }
}

  const getConnectionStatusDisplay = () => {
    switch (connectionStatus) {
      case "connecting":
        return <div className="text-yellow-600 text-sm px-4 py-2 bg-yellow-50 border-l-4 border-yellow-400">Connecting to server...</div>;
      case "connected":
        return null; // Don't show anything when connected
      case "disconnected":
        return <div className="text-orange-600 text-sm px-4 py-2 bg-orange-50 border-l-4 border-orange-400">Disconnected. Attempting to reconnect...</div>;
      case "error":
        return <div className="text-red-600 text-sm px-4 py-2 bg-red-50 border-l-4 border-red-400">Connection failed. Please refresh the page.</div>;
      default:
        return null;
    }
  };

  return (
    <div className="flex flex-col min-w-0 h-dvh bg-background">
      <Header/>
      {getConnectionStatusDisplay()}
      <div className="flex flex-col min-w-0 gap-6 flex-1 overflow-y-scroll pt-4" ref={messagesContainerRef}>
        {messages.length == 0 && <Overview />}
        {messages.map((message, index) => (
          <PreviewMessage key={index} message={message} />
        ))}
        {isLoading && <ThinkingMessage />}
        <div ref={messagesEndRef} className="shrink-0 min-w-[24px] min-h-[24px]"/>
      </div>
      <div className="flex mx-auto px-4 bg-background pb-4 md:pb-6 gap-2 w-full md:max-w-3xl">
        <ChatInput  
          question={question}
          setQuestion={setQuestion}
          onSubmit={handleSubmit}
          isLoading={isLoading || connectionStatus !== "connected"}
        />
      </div>
    </div>
  );
};