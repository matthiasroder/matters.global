import { ThemeToggle } from "./theme-toggle";
import { APP_SETTINGS } from "@/config";

export const Header = () => {
  return (
    <>
      <header className="flex items-center justify-between px-2 sm:px-4 py-2 bg-background text-black dark:text-white w-full">
        <div className="flex items-center space-x-1 sm:space-x-2">
          <h1 className="text-xl font-bold">{APP_SETTINGS.title}</h1>
        </div>
        <div className="flex items-center space-x-1 sm:space-x-2">
          <ThemeToggle />
        </div>
      </header>
    </>
  );
};