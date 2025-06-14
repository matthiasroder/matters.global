import { motion } from 'framer-motion';
import { MessageCircle, BotIcon, ListChecks } from 'lucide-react';
import { APP_SETTINGS } from '@/config';

export const Overview = () => {
  return (
    <>
    <motion.div
      key="overview"
      className="max-w-3xl mx-auto md:mt-10"
      initial={{ opacity: 0, scale: 0.98 }}
      animate={{ opacity: 1, scale: 1 }}
      exit={{ opacity: 0, scale: 0.98 }}
      transition={{ delay: 0.5 }}
    >
      <div className="rounded-xl p-6 flex flex-col gap-8 leading-relaxed text-center max-w-xl">
        <p className="flex flex-row justify-center gap-4 items-center">
          <BotIcon size={44}/>
          <span>+</span>
          <ListChecks size={44}/>
          <span>+</span>
          <MessageCircle size={44}/>
        </p>
        <h2 className="text-2xl font-bold">{APP_SETTINGS.title}</h2>
        <p className="text-lg">{APP_SETTINGS.description}</p>

        <div className="text-left mt-4">
          <h3 className="font-semibold mb-2">I can help you with:</h3>
          <ul className="list-disc pl-6 space-y-1">
            <li>Creating and tracking problems that matter to you</li>
            <li>Breaking down problems into specific conditions</li>
            <li>Finding connections between related problems</li>
            <li>Checking for similar existing problems</li>
            <li>Tracking progress on problem resolution</li>
          </ul>

          <p className="mt-6 text-center italic">
            Start by describing a problem you'd like to track or ask me to list your existing problems.
          </p>
        </div>
      </div>
    </motion.div>
    </>
  );
};
